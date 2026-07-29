# v6 失败归因、v6.2 短层距归档与 v6.3 正式间距结果

## v6.3 正式 1.50 mm self-imaging smoke（2026-07-29）

v6.3 是可搭建实验台合同：正式 1.50 mm、2+2 phase、identity OLS、固定曝光，显式接续通过的
逐类逐方向电子 PCA scale，但相位零初始化。其物理 admission 证明随机码仍到达 CMOS（axis/uniform
fraction `0.2531/0.4603/0.4814`）；无更新 semantic tangent 也证明 carrier 与 phase 存在局部
改善语义的方向。唯一 400-step smoke 后语义完全恢复，但 axis fraction 变为 `0.2203 < 0.25`。

这将根因收窄为：在当前 2+2、1.50 mm、固定曝光和此 PCA command 分布下，现有 self-imaging/
classification 优化会优先形成类别稳定模板，并牺牲最严格的轴向扰动传输。两组 uniform 仍高于门，
故不是随机码全面消失；同时五组梯度、更新、饱和和显存均正常，不能归因于训练停滞或 CUDA。
该有效硬门失败停止 v6.3；禁止重跑、延长或调 gain/LR/loss。后续若继续，只能另立物理架构合同，
例如明确设计层间 1:1 relay，或转向单片多层器件。

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

## 首次 v6.2 target admission（2026-07-29，已 supersede）

使用当前内容绑定 semantic checkpoint、冻结 evaluator 和本地 MNIST 重新执行电子端门。
固定 20 次二分把语义安全 scale 推到硬上限 `0.19999986`；三个固定面板的 accuracy
为 `1.0/1.0/0.9882`，每类 recall 均满足门，但 `A²` code fraction 仅为
`0.08990/0.22123/0.23989`；因此 `conditional_short_spacing_target_admission`
为 `failed`。该失败发生在光路之前，未创建 optimizer、未生成初始化 checkpoint，故
不允许继续 optical admission 或 CUDA smoke。最新运行目录为
`outputs/20260729_v2/runs/20260729_181916_504477_diagnostic_v2_pca_self_imaging_transport_finite_crop_identity_short_spacing/`。

该证据的全局scale、0.20上限、错误失败原因和CLI零退出码均违背预注册合同，分类为
`invalid_harness_evidence`；原文件保持只读。

## v6.2.1 电子与短层距光学准入（2026-07-29）

修复为每类每方向独立的 `(0.05,0.50)` scale 后，电子target admission通过：axis、
uniform-1234、uniform-5678 的 accuracy均为 `1.0`，`A²` code fraction为
`0.3095/0.5066/0.5387`，每类recall和八方向响应均通过。

随后零phase的0.25 mm CUDA光学准入为有效失败。冻结全局 `κ=0.997967`，无饱和、
trace完整；但相对于1.50 mm，Encoder layer 2 code retention仅提升 axis `27.4%`、
uniform-1234 `16.0%`、uniform-5678 `14.3%`。三面板均需至少25%的合同未满足，故
分类 `short_spacing_optical_retention_insufficient`。不得进入observability、400-step smoke、
正式训练或数据集导出。运行目录：
`outputs/20260729_v2/runs/20260729_213431_379723_diagnostic_v2_pca_self_imaging_transport_finite_crop_identity_short_spacing/`。
