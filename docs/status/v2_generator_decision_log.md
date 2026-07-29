# v2 generator 决策记录

本记录只总结当前teacher-free全光路生成主线的结构决策，不授权任何训练或数据集导出。

## 已停止的结构

| 候选 | 停止原因 |
| --- | --- |
| Identity v2 `Wz/Wy` | 能保持类别语义，但通过跨维系数映射压缩随机码。 |
| Wy=0 / v3共享或类条件模式 | 多样性恢复后，非零码语义与类别5/7冲突。 |
| v4 parity stabilizer | 只读切空间能改善margin，但会破坏特征多样性；奇偶抵消改善不足。 |
| v5四锚点局部模式 | 四簇平均收益仅0.1691，最低类别0.1243；数字1有156张小簇，不支持等权硬路由。 |

## 当前有效边界

上述失败均不否定宏观物理链路：

```text
label + code → amplitude SLM → Encoder → identity OLS → Decoder → CMOS
```

它们否定的是电子端入口的参数化和数据几何假设。失败evidence保持不可变，只能用于下一次诊断定位，不能授权训练、自动评估或导出。

## 当前 v6 决策门

连续 PCA 容量审计已通过，但安全尺度物理传输审计以 `safe_scale_transport_insufficient` 失败：语义安全范围与
可验收 raw-CMOS code response 没有重叠。当前候选为 v6 `conditional_pca_transport_codesign_v6`：它明确共同优化
载波、20 cycles/mm 固定低通 PCA 模式、有界尺度和四层相位；不是 gain sweep，也不复用旧 generator 权重。
v6 只可按 `admission → observability → 一次400-step smoke → 用户5 epoch` 推进。任一门失败即转入宏观光路
生成可行性复审，禁止 v7、降门限、重复 smoke 或延长相同训练。

## v6 smoke 与宏观可行性复审结果（2026-07-29）

v6 的 admission/observability 通过，但唯一 400-step smoke 真实失败：语义 accuracy 达到 `1.0`，
而三个面板 code fraction 仅 `0.0106/0.0325/0.0354`，最低 feature ratio 为 `0.288/0.399/0.442`。
分类 loss 和 evaluator 特征损失下降没有转化为足够 raw-CMOS 物理变化。固定 `lowpass_mask` 被错误纳入旧
updates gate，已由只读复核工具排除；五组可训练参数仍均有有限非零更新，因此该缺陷不改变模型失败结论。

流式宏观审计在 band-limited SLM、SLM+共享 phase 和 identity-phase reference 三个局部 counterfactual
空间均未满足联合门，结果为 `local_macro_generation_unreachable`。该证据只表示当前 v6 初始化、20 cycles/mm
输入带宽、2+2 phase 和固定曝光合同的局部可达性不足，不授权创建 v7 或继续训练。

当前停止条件：必须先完成宏观物理合同复审，明确是否允许改变 phase 初始化、光学层容量/间距/孔径或质量门；
在得到明确合同前，禁止正式训练、自动评估、candidate 导出、防记忆、utility 和最终准入。

## v6.2 短层距自成像（2026-07-29）

v6 失败归因为 1.50 mm 层距下相位职责与随机码传输目标错位，而不是 MNIST 容量或注入失败。
真实 MNIST 同一 ANOVA 指标的 camera fraction 约 0.710，故不降低 0.25 门限。新 diagnostic
架构 `conditional_pca_self_imaging_transport_v6_2` 将层距固定为 0.25 mm，并让 raw CMOS
对同 code 的 `κA²` 与 delta 变化负责；target 分支 detach，禁止 teacher/paired-MSE。

旧 v6 证据不能迁移。短层距链必须重新通过 target admission、optical admission、observability
和唯一 400-step smoke；失败后停止，不创建 v7。

### v6.2.1 optical admission结果

修复每类每方向scale与CLI返回码后，电子target通过；但0.25 mm零phase的Encoder layer 2
retention相对1.50 mm只在axis达到`+27.4%`，两组uniform仅`+16.0%/+14.3%`。因三面板
`+25%`是预注册物理门，`conditional_short_spacing_optical_admission`有效失败。短层距自成像
训练尚未开始，也不得通过调loss、重复smoke或降低门限绕过该证据。
