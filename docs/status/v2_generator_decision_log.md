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
