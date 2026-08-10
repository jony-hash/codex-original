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

### Current training schedule

Current identity training uses the `balanced_min_class_exhaustion` iterator with
the configured logical batch size and seed.  The schedule contract is therefore
split-derived: for the formal MNIST split it produces 984 steps per epoch and
4,920 steps over five epochs.  A current final training checkpoint must carry
`BALANCED_TRAINING_SCHEDULE_CONTRACT`, and its history, manifest and training
evidence must agree on the per-epoch and total step counts.  The older 5,500-step
contract remains readable only for historical/initialization compatibility; it
cannot authorize current evaluation.  Metadata-only correction evidence for an
old run is historical and is not a replacement for a newly executed current run.

## 当前执行顺序

```text
evaluator qualification
→ generator-only injection admission
→ no-update optical observability
→ no-update control isolation
→ 根据 control-isolation 分类选择 semantic-localization 或 controllability smoke
→ 通过 controllability smoke 或 semantic-recovery evidence 授权 5-epoch joint training
→ raw-CMOS evaluation
→ manual review
→ candidate dataset export
→ memorization and utility
→ accepted manifest
```

公开命令继续使用通用的 `student train/sample/evaluate`，但由
`architecture` 严格分派。identity 主线的 evidence 必须逐项绑定 system、config、
evaluator、MNIST split、initialization checkpoint 和 action；校验失败时不能创建
optimizer 或 run 工件。正式 identity 当前只注册 current handler；未注册的 identity
命令（例如 legacy preflight）直接 fail-closed，不会回退到 diagnostic 或 legacy。

训练授权必须且只能提供一种前置证据：通过的 `controllability_smoke`（100-step
合同、控制 metric contract、初始化 checkpoint/state 和 schedule hash 完整绑定），
或通过的 `conditional_semantic_recovery_smoke`（400-step 合同）。两条路径均从原始
immutable initialization checkpoint 构造 bundle，smoke/recovery checkpoint 不能作为
正式训练 warm start。当前命令 `audit conditional-semantic-localization`、
`student admit-dataset` 和 `student audit-memorization` 均由 current 实现处理；
`audit memorization` 是保留的兼容别名。

候选数据集始终写入：

```text
outputs/YYYYMMDD_v2/runs/<run>/conditional_mnist_dataset/
```

其 manifest 状态为 `candidate`。只有 generation admission、memorization 和 utility
全部通过，才原子写入 `accepted_manifest.json`。这不等同于硬件加工或 scientific
promotion。

## 初始化、加载与证据链

current identity 的唯一 bundle 构造入口在构造任何参数前读取配置中的
`initialization_seed`，并在局部 RNG 作用域内完成初始化。初始化 checkpoint 的
manifest 必须同时满足 seed、`state_content_hash` 和
`initialization_state_hash` 一致；restore 先验证这些关系，再把同一配置 seed
用于构造 bundle。

训练 evidence 的现有 `input_hashes` 绑定初始化 checkpoint 文件摘要、初始化
state hash、最终 checkpoint 摘要、配置和数据 split。evaluation、dataset 和 recovery
消费前必须验证最终 manifest 与训练 evidence 的初始化 state hash；任何 seed、文件、
manifest 或 evidence 输入摘要被篡改都必须 fail-closed。此链路只证明工程来源和
更新真实性，不替代 CUDA、硬件或科学 acceptance。

Current evidence 还受 runtime-tree 来源约束：同一消费操作的上游记录必须使用相同的
`git_commit`，且所有记录的 `dirty_patch_hash` 必须为 `None`。记录的提交必须是当前
`HEAD` 的祖先，并且从该提交到当前 `HEAD` 的以下路径不得有差异：
`src/ldm_project/v2/`、`configs/v2/current/` 和 `pyproject.toml`。因此 docs/tests-only
后续提交可以继续消费既有 evidence；runtime、current 配置或项目依赖变化必须重新签发
整条 evidence 链。Git 不可用、提交无效或比较失败时，current 入口在加载 checkpoint、
创建 optimizer、写入 run 工件之前 fail-closed。

所有 current、student、teacher、evidence、diagnostic 和 research v2 artifact 都通过
`safe_torch_load` 反序列化，默认 `weights_only=True`，并在有期望摘要时先校验文件。
唯一的 `weights_only=False` 位于 legacy 私有可信加载边界；不提供 current CLI 绕过开关。

current 与 diagnostic 共享中立的 admission/recovery 工作流绑定，但依赖方向仍是单向的：
current 不导入 diagnostic 或 legacy，v2 parser 也不加载执行 runtime。动态绑定是兼容实现
边界，不构成 current 对候选架构的运行时依赖。

## 旧 v2 内容的状态

旧 teacher/prototype/capacity、global-amplitude、direct-template 和 LightGen 保留在
显式 `legacy` 或 `diagnostic` 兼容命名空间，只用于解释和复现既往证据，不能进入
identity 正式命令的导入面，也不能签发 acceptance。`configs/v2/current/` 只包含
identity 配置；旧配置路径通过
[`path_migration_manifest.yaml`](../../configs/v2/path_migration_manifest.yaml) 精确映射，
不会猜测相似路径，也不会静默迁移 evidence 或 checkpoint。

正式 CLI 先以只依赖 `argparse` 的 schema 解析命令和验证配置，再通过
`module:function + 调用类型` 描述惰性分派。identity 命令只加载 current 的 runtime、
admission/recovery、training、evaluation 或 dataset 职责模块，以及 formal optics、
evidence 和 evaluator；`identity_runtime` 与 `conditional_runtime` 都仅保留为惰性兼容
facade，不进入正式 identity 的导入闭包。teacher/prototype/direct-template 归 legacy，
class/parity/multi-anchor 和 v6–v7.2 归 diagnostic，均不能签发 acceptance。

为保留历史复现，旧的 `application.conditional_*` Python 路径仍然可导入，但它们只是
惰性 shim，实际 v6–v7.2 实现均位于 `application.diagnostic`。正式 current 与
diagnostic 仅共享不含架构分派或 acceptance 判定的运行时辅助函数；current 不会反向
导入 diagnostic 或 legacy。旧 `legacy.cli_impl` 同样转发到唯一的公共 parser/dispatcher，
不再维护第二份 CLI 定义。

当前 identity 尚未产生可 promotion 的正式 checkpoint。旧 smoke、prototype 和诊断结果
保持历史状态；若 split、config 或 evidence context 变化，必须从 evaluator qualification
重新签发。

## v5 多锚点候选（已拒绝，diagnostic-only）

`conditional_multi_anchor_local_modes_v5` 保留完整宏观光路，但将电子端入口改为每类4个振幅锚点和每个锚点6个局部模式。前两维code只选择锚点，后六维code只调制所选锚点；没有Wz/Wy、软件decoder或光路后latent。无更新容量审计未通过，因此该候选不属于当前正式运行时；不得继续运行v5 admission、observability或smoke。旧v1-v4 checkpoint/evidence不能部分迁移。

连续PCA模式目前仅有diagnostic-only容量审计接口，尚未注册为正式architecture，也没有训练checkpoint或推理合同。
