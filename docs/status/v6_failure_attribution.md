# v6 失败归因与 v6.2 短层距方案

## 结论

2026-07-29 的 v6 `conditional_pca_transport_codesign_v6` 唯一 400-step smoke
是有效失败：语义 accuracy 为 1.0、饱和约 0.047，但三个面板 raw-CMOS code
fraction 仅 0.0106/0.0325/0.0354，最低特征距离比为 0.288/0.399/0.442。
五组可训练参数均有更新，故不是梯度、显存或 optimizer 缺失；camera loss 和
同 code 物理传输之间存在目标错位。

宏观局部审计 `local_macro_generation_unreachable` 只针对 1.50 mm、当前相位
盆地和已注册切空间，不是任意光路的数学不可能性证明。所有旧 evidence、checkpoint
和 diagnostic final 保持只读，不授权继续训练或导出。

## v6.2 合同

新 diagnostic 架构为 `conditional_pca_self_imaging_transport_v6_2`，层间距
固定为 0.25 mm，仍是 `label+8D code → amplitude SLM → 2+2 Encoder →
identity OLS → 2+2 Decoder → CMOS`。generator 仍为低通 PCA 模式和有界 scale，
新增同 code 自成像目标 `Traw = κ A²` 及 delta 传输损失；target 分支 detach，
不引入 teacher、VAE、DDPM、软件 decoder 或逐样本归一化。

执行顺序固定为 target admission → optical admission → observability → 唯一
400-step short-spacing smoke。任一 gate 失败即停止，不调整 gain/LR/loss/层距，
正式 5 epoch、5000 validation、60000 导出由用户在 smoke 通过后运行。

## 首次 v6.2 target admission（2026-07-29）

使用当前内容绑定 semantic checkpoint、冻结 evaluator 和本地 MNIST 重新执行电子端门。
固定 20 次二分把语义安全 scale 推到硬上限 `0.19999986`；三个固定面板的 accuracy
为 `1.0/1.0/0.9882`，每类 recall 均满足门，但 `A²` code fraction 仅为
`0.08990/0.22123/0.23989`；因此 `conditional_short_spacing_target_admission`
为 `failed`。该失败发生在光路之前，未创建 optimizer、未生成初始化 checkpoint，故
不允许继续 optical admission 或 CUDA smoke。最新运行目录为
`outputs/20260729_v2/runs/20260729_181916_504477_diagnostic_v2_pca_self_imaging_transport_finite_crop_identity_short_spacing/`。
