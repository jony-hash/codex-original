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

`rewrite/v2` 从 v79 快照 `e21dba1` 线性分出，当前入口为 `python -m ldm_project.v2`；它不替换 `train.py` 或 v1 默认行为。v2 以类型化 physical field、严格配置/manifest、独立光学 reference、prototype/capacity evidence gate 和受约束 code basis 重写正式主链。

已完成代码、CPU diagnostic 竖切与单样本 CUDA forward/capacity smoke；这些只说明执行和数值路径，不构成 scientific acceptance。尚未完成正式 capacity evidence、长 CUDA 训练、十类人工 raw-CMOS 审查、bootstrap paired-teacher advantage 或 promotion。

## 下一允许动作

v2 的正式执行顺序固定为：内容寻址 teacher manifest → `student calibrate-prototypes` → `audit capacity` → `student train` → `student sample/evaluate`。prototype 或 capacity gate 失败时，后续训练、test、导出和 promotion 自动关闭。

在具备 clean commit、匹配 teacher/data/checkpoint/calibration hash、固定曝光与人工审查输入后，由用户在 CUDA 上依序运行 v2 的 prototype 与 capacity gate。未取得这些证据前，不改写 v1 科学结论，不将 v2 设为默认入口，也不签发 acceptance。
