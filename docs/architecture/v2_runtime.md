# v2 Teacher-Free Conditional Runtime

v2 当前主线是 `conditional_identity_logit_amplitude_v2`，入口仍为：

```bash
python -m ldm_project.v2 --help
```

它不替换 v1 的 `train.py`，也不导入 LDM teacher。正式物理链保持不变：

```text
label + 8D code
→ identity-anchored electronic amplitude generator
→ amplitude SLM
→ coherent 4f relay
→ 2-layer finite-crop phase-only encoder
→ complex identity OLS
→ 2-layer finite-crop phase-only decoder
→ B=0 backend relay
→ raw CMOS intensity
```

## 当前配置与边界

当前配置为
[`formal_identity_logit_amplitude_finite_crop_identity.yaml`](../../configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml)。
它锁定 532 nm、200×200 SLM、100×100 @ 2 µm metasurface、2+2 phase-only、
1.50 mm spacing、finite-crop、identity OLS、固定曝光和 8D code。

generator 的 identity anchor 为：

```text
h(z) = SiLU(Wz · z)
r(z,y) = Wy · h(z)
c(z,y) = tanh(z + r(z,y))
A(z,y) = sigmoid(By + 0.5 × Σk ck(z,y) P̂k)
```

MNIST 只用于训练和验收。冻结 evaluator 不进入最终光学推理；MNIST 到 SLM
的 carrier 初始化是电子端实值输入准备，不是传播过程中的 complex resize。

## 当前执行顺序

```text
evaluator qualification
→ generator-only injection admission
→ no-update optical observability
→ no-update control isolation
→ 唯一一次 100-step controllability smoke
→ 用户运行 5-epoch joint training
→ raw-CMOS evaluation
→ manual review
→ candidate dataset export
→ memorization and utility
→ accepted manifest
```

公开命令继续使用通用的 `student train/sample/evaluate`，但由
`architecture` 严格分派。identity 主线的 evidence 必须逐项绑定 system、config、
evaluator、MNIST split、initialization checkpoint 和 action；校验失败时不能创建
optimizer 或 run 工件。

候选数据集始终写入：

```text
outputs/YYYYMMDD_v2/runs/<run>/conditional_mnist_dataset/
```

其 manifest 状态为 `candidate`。只有 generation admission、memorization 和 utility
全部通过，才原子写入 `accepted_manifest.json`。这不等同于硬件加工或 scientific
promotion。

## 旧 v2 内容的状态

旧 teacher/prototype/capacity、global-amplitude、direct-template 和 LightGen 文本已
移入 archive，只用于解释既往证据，不能作为当前命令来源。旧配置路径通过
[`path_migration_manifest.yaml`](../../configs/v2/path_migration_manifest.yaml) 精确映射，
不会猜测相似路径，也不会静默迁移 evidence 或 checkpoint。

当前 identity 尚未产生可 promotion 的正式 checkpoint。旧 smoke、prototype 和诊断结果
保持历史状态；若 split、config 或 evidence context 变化，必须从 evaluator qualification
重新签发。
