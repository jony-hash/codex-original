# 当前主线交接：v79 冻结与 v2 旁路（2026-07-22）

本文件只保存当前状态、停止规则和下一允许动作。v68–v79 的旧交接正文已完整、只读保存在 [v68–v79 原始主线交接归档](current_handoff_v68_v79_archive.md)，按阶段查阅的摘要、指标和工件索引见 [实验与验收历史](experiment_history.md)；稳定架构合同见 [D2NN 单次曝光蒸馏主线](single_shot_ldm_distillation.md)；分支与回档点见 [Git 分支治理指南](../git_branch_management.md)。

## 当前正式合同

```text
laser -> amplitude SLM -> coherent 4f relay -> encoder D2NN (2 layers)
      -> complex identity OLS -> decoder D2NN (2 layers)
      -> finite-conjugate coherent relay -> raw CMOS intensity
```

- `lambda=532 nm`；无 fiber；OLS 为严格复线性恒等映射。
- SLM/CMOS 为 `200×200 @ 2.5 µm`（`500 µm` FOV）。encoder/decoder 各为 `100×100 @ 2 µm`（`200 µm` FOV）、2 层 phase-only、层间距 `1.50 mm`。
- 前端为完整 FOV 的 `M=-0.4` 相干 4f relay；后端为 `M=-2.5` 的有限共轭 relay。正式 D2NN 边界为 propagation-after-crop 的 `finite_crop`，不是材料不透明性声明。
- raw CMOS、loss view 和 display view 不能混用；最终可读性只看锁定曝光的 raw CMOS，不接受增强预览或软件读出替代。

## v1 研究线的冻结结论

v68 A2 曾通过 prototype raw/readability 门，v68.6 B 与 v69 也保留了可读 prototype；但它们的 raw-camera code fraction 分别为 `7.14e-08` 和 `0.0002712`，远低于 `0.25`。v70–v77 将衰减定位为冻结 D2NN 的 phase-to-intensity/readout 权衡，未产生可部署的训练授权。

v79 以固定 label-only normalization 对八维 supported basis 做 CPU 上界：frozen replay 的 physical-SLM code factor 为 `6.03e-7`，deterministic capacity arm 为 `0.05455 < 0.25`，且 clipping `0.20547 > 0.08`。有效结论为 `label-fixed-supported-basis-insufficient`：该 basis 家族不足以证明 code transport；learned code transport 仍未证明，且不存在可 promotion 的 student checkpoint。

因此冻结 v1 semantic-supported basis、residual＋phase 联合训练、normalization/gain/amplitude/basis sweep、test、导出和 Formal promotion。不得把 prototype readability、teacher-camera 相似度或离线 software readout 表述为最终单次曝光生成能力。

## v2 旁路状态

`rewrite/v2` 从 v79 快照 `e21dba1` 线性分出，当前入口为 `python -m ldm_project.v2`；它不替换 `train.py` 或 v1 默认行为。v2 以类型化 physical field、严格配置/manifest、独立光学 reference、prototype/capacity evidence gate 和受约束 code basis 重写正式主链。其执行链已固定为 teacher pair/target manifest → calibration measurement → content-bound prototype review/admission → frozen capacity → guarded training → raw-CMOS evaluate/review。

已完成代码和 CPU diagnostic 全链竖切；该竖切使用 fake teacher pair、内部 physical-DCT capacity 和单步训练，所有输出强制为 diagnostic。v2 已具备 8 维/每类 100 个 teacher export 合同、本地 128-per-label clean target 导出、流式 CUDA 执行、固定曝光 review 工件及无更新 `student preflight`。正式 8D teacher bundle 已完成内容审计；但首次 formal prototype calibration 在 `d7958bc` 上有效失败：统一 `0.5` amplitude template、零 phase、5 个实际 optimizer step 的 raw corr/dice/accuracy 为 `0.34985/0.49562/0.10`。该 evidence 保留为有效失败，不是 harness-invalid，也不能直接推出 finite-crop 物理不可达。只读 probe 显示 `sqrt(clean intensity)` amplitude 初始化可达到 corr/dice `0.77961/0.76542`，但 accuracy `0.40`；50-step MSE 或 gate-aligned probe 未改善 accuracy。下一门是 prototype reachability broad isolation，未完成前不重跑 formal calibration、capacity 或 student training。

## 下一允许动作

v2 的正式执行顺序固定为：clean commit 上的 `student preflight` → local clean-target manifest → teacher pair manifest → 8D atomic target export/target manifest → `student calibrate-prototypes` → `student admit-prototypes` → `audit capacity` → `student train` → `student evaluate` → `student admit-evaluation`。prototype calibration 的自动 corr/dice/accuracy/saturation/finite/gradient/memory/epoch gate 任一失败会直接停止，不进入人工审阅；CPU diagnostic evidence 只能显式授权 diagnostic smoke，不能跨入 formal 链。preflight 通过只授权用户进入 prototype gate，不构成 capacity、科学或 promotion acceptance。

在具备 clean commit、匹配 teacher/data/checkpoint/calibration hash、固定曝光与人工审查输入后，由用户在 CUDA 上依序运行 v2 的 prototype 与 capacity gate。未取得这些证据前，不改写 v1 科学结论，不将 v2 设为默认入口，也不签发 acceptance。
