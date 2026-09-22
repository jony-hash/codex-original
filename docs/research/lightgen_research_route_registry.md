# LightGen 研究路线与 distribution 注册表

更新时间：2026-09-17  
注册表版本：`LIGHTGEN_ROUTE_AND_DISTRIBUTION_REGISTRY_V1`

文档状态：`DRAFT_NOT_AUTHORITY`（当前未跟踪草稿）  
拟定文档角色：`STABLE_NAVIGATION`；在纳入 Git 前不属于项目 authority chain。

> 本文件是项目路线与术语注册表的工作树草稿。本次内容只整理路线身份、分布对象、目标函数
> 和 claim 边界；不新增科学证据，不改写历史报告，不授权训练、CUDA、采样、生成、物理验证
> 或 promotion。当前 latest、blocker、current next action 和执行授权仍以
> `docs/CURRENT_STATUS.md`、`docs/status/current_handoff.md` 与根/作用域合同为准。

## 1. 权威与边界

- 稳定目标与长期边界以 [`PROJECT_MEMORY.md`](../PROJECT_MEMORY.md) 为准；当前数值、
  阻塞和执行边界以 [`CURRENT_STATUS.md`](../CURRENT_STATUS.md) 与
  [`current_handoff.md`](../status/current_handoff.md) 为准；日期化实验事实以
  [`experiment_history.md`](../status/experiment_history.md) 为准。
- LightGen 主线的输入/OLS 边界由
  [`lightgen_2d_semantic_generation_input_ols_alignment.py`](../../src/ldm_project/v2/application/research/lightgen_2d_semantic_generation_input_ols_alignment.py)
  定义：真实图像 → 振幅 SLM → optical encoder → `z∈C^100` 复 OLS 系数 →
  `W_OLS` 模式重发 → optical generator → raw CMOS。系数空间 `T=I_100` 不等于整幅
  光场 identity bypass。
- 正式 v2 的 `label + 8D` / identity OLS 由
  [`v2_runtime.md`](../architecture/v2_runtime.md) 与
  [`identity_generator.py`](../../src/ldm_project/v2/student/identity_generator.py)
  定义，是 compatibility baseline；它不定义 M0，也不能从形式 runtime 推导
  LightGen 论文对齐或 generation readiness。
- `SOURCE_AUTHORITIES` 只指向项目内代码、架构、状态和历史入口；它不把 hash、计数或
  讨论文本单独提升为科学 gate。

## 2. 术语注册

以下术语必须带限定轴使用；裸写“distribution”不能作为路线、损失或 claim。

| 术语 | 注册定义 | 不得混同 |
| --- | --- | --- |
| `DATA_DIST` | 路线实际读取的输入数据 `(x,y)` 在声明 split/角色下的经验分布。 | 不等于 OLS latent 或 evaluator feature 分布。 |
| `LATENT_EMPIRICAL_DIST` | 在明确的物理 latent 空间中，由真实 encoder/观测系数 bank 得到的有限经验分布。 | 不等于先验，也不等于可访问 manifold。 |
| `LATENT_PRIOR_DIST` | 路线声明用来采样、拟合或约束 latent 的目标概率律。没有声明时必须写 `NONE` 或 `UNKNOWN`。 | 不得把 N3 候选或 S1 的训练 target 自动继承给 M0。 |
| `LATENT_SAMPLE_DIST` | 已实际物化的有限 latent sample bank 的分布；它只描述样本产物，不自动证明光学生成。 | 不等于 CMOS 图像生成结果或 population law。 |
| `SEMANTIC_FEATURE_DIST` | 明确 evaluator/task feature 空间中的经验分布，例如 label、prototype、embedding 或 transfer margin。 | 不等于 `LATENT_EMPIRICAL_DIST`，也不等于图像像素分布。 |
| `OPTICAL_POWER_DIST` | 端口、光学平面或样本上的功率/能量/传输观测分布，必须声明观测位置和归一化。 | 不等于 latent amplitude 的统计律。 |
| `MANIFOLD` | 可访问状态的 support、几何结构或局部方向集合。 | manifold 不是 probability mass。 |
| `DISTRIBUTION` | 在已声明空间上的 probability mass。 | 不能用来含糊代指 manifold、feature、power 或样本 bank。 |
| `SEMANTIC` | 与任务标签或可辨识内容有关的信息。 | 语义 margin、图像 loss、latent geometry 三者不能互相代替。 |

## 3. 状态与角色规则

- `ROLE` 是路线的主要用途且每条路线只能有一个：`MAINLINE`、`SUPPORT_BRANCH`、
  `BASELINE` 或 `HISTORICAL_CLOSED`。
- 只有 `M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION` 是 `MAINLINE`。支持分支可以保留
  候选组件或负结果，但不能自动打开 M0、Stage B、generation 或 promotion。
- `GENERATION_EXECUTED` 只在真实路线的光学链产生并完成声明验收的 CMOS 输出时才写
  `YES`。只拟合 latent、物化 sample bank、运行 evaluator、做 cyclic transfer 或
  调用历史 `generate_from_prior` 都不能写成当前 generation evidence。
- `MMD` 是分布损失；`BOGT` 是另一个需有明确 paper/implementation authority 的方法
  名称。S1 不得写成 BOGT，H1 的 scaled-Chen 实现也不得写成已复现论文 BOGT。
- `Strict ReLU90` 只是在 optical facet 上的算法性 `I' = ReLU(I-τ)` surrogate；它
  不是 latent probability model，也没有定义 `LATENT_PRIOR_DIST`。

## 4. 项目树

```text
MAINLINE
└── M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION
    proposed Chen-inspired BOGT-on-complex-OLS route; not executed

SUPPORT_BRANCHES
├── S1_COMPLEX_OLS_V1_FULL_COMPLEX_MMD
├── S1_COMPLEX_OLS_V2_NORMALIZED_MMD
├── S2_SEMANTIC_PATH_DIAGNOSTIC_V3_V11
│   ├── V3 objective correction → V4 fixed-OLS recoverability
│   ├── V5 direct complex-OLS semantic alignment
│   ├── V6 proto-only → V7 transfer → V8 anchor+transfer
│   ├── V9 joint remapping → V10 invalid attempt
│   ├── V10.1 rank-9 remap → V11 semantic/reconstruction compatibility
│   └── all are bounded diagnostic subroutes, not generation routes
├── S3_N3_LATENT_DISTRIBUTION_PRODUCER_FAMILY
└── S4_ABCD_ARCHITECTURE_NONLINEAR_ABLATION
    ├── A/B/C/D 2L/4L × linear/ideal-ReLU facet
    ├── S4_N5_TRUE_OPTICAL_DEPTH_COMPARISON
    ├── S4_N6_NONLINEAR_FACET_ACTIVATION_CANARY
    ├── S4_4L_LINEAR_OPTIMIZATION_CEILING_CONTINUATION
    └── S4_STRICT_RELU90_TRAIN_CALIBRATED

BASELINES
└── B0_FORMAL_LABEL8D_IDENTITY_OLS_COMPATIBILITY

HISTORICAL_CLOSED
├── H1_SCALED_CHEN_BOGT_RECONSTRUCTION_REPAIR
│   └── legacy S1–S3 controller / rank-16 readout remain historical siblings
└── H2_V1_LDM_TEACHER_REPRODUCTION
```

## 5. 统一路线记录

每张路线卡都使用同一组字段。S2 和 S4 的子路线表登记了独立 `ROUTE_ID`；除覆盖项
外，子路线继承其父卡的全部字段，因此不会因为压缩重复文本而丢失统一 schema。

### 5.1 M0：LightGen 输入与 OLS 对齐的 proposed mainline

```text
ROUTE_ID: M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION
ROLE: MAINLINE
STATUS: PROPOSED_NOT_EXECUTED
PARENT: LightGen input/OLS alignment definition and current research objective
SCIENTIFIC_QUESTION: Can a project-declared Chen-inspired aggregate BOGT objective improve semantic optical generation at the complex OLS boundary while preserving raw CMOS reconstruction and optical-power constraints?
PHYSICAL_LATENT: z∈C^100 complex modal OLS coefficients at the 10×10 interface; real-image encoding and future external z entry share the W_OLS/generator boundary
STATISTICAL_VARIABLE: UNKNOWN; must be declared on OLS coefficient, amplitude/power, phase, or a justified quotient space before execution
DATA_DIST: UNKNOWN for a new M0 contract; candidate input roles are declared MNIST TRAIN/DEV image-label samples
LATENT_EMPIRICAL_DIST: UNKNOWN; must be measured from real encoded z at the OLS boundary
LATENT_PRIOR_DIST: UNKNOWN; no S1, S2, S3, B0, or H1 law may be inherited automatically
DISTRIBUTION_OBJECTIVE: PLANNED_ONLY; Chen-inspired aggregate constraint, with statistic, target, axis, and weight still unknown
SEMANTIC_OBJECTIVE: PLANNED; task-label/recognizable content in raw CMOS, with evaluator and loss still unknown
IMAGE_OBJECTIVE: PLANNED; raw CMOS intensity reconstruction constraint, exact contract still unknown
OPTICAL_POWER_OBJECTIVE: PLANNED; optical-plane/operating-window power term, observable and weight still unknown
TRAINABLE_PARAMETERS: UNKNOWN; a new bounded contract is required before any optimizer exists
SAMPLING_MECHANISM: UNKNOWN; future external z producer must be specified and must share the declared W_OLS and generator interface
GENERATION_EXECUTED: NO_ACCEPTED_M0_GENERATION
SUPPORTED_CLAIM: The unique current research identity and a proposed route/question only
FORBIDDEN_CLAIM: Successful LightGen generation, exact paper BOGT reproduction, current Scaled-Chen implementation, N3 as a complete generator, or hardware readiness
CURRENT_NEXT_ACTION: No scientific action in this docs-only task; if resumed, define one bounded M0 contract before any run
SOURCE_AUTHORITIES: lightgen_2d_semantic_generation_input_ols_alignment.py; PROJECT_MEMORY.md; CURRENT_STATUS.md; current_handoff.md; LightGen registered supplement
```

### 5.2 S1：complex-OLS MMD baseline family

#### S1-V1：raw/full-complex MMD

```text
ROUTE_ID: S1_COMPLEX_OLS_V1_FULL_COMPLEX_MMD
ROLE: SUPPORT_BRANCH
STATUS: CLOSED_TRAINED_NO_GENERATION_ACCEPTANCE
PARENT: M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION
SCIENTIFIC_QUESTION: Does a full-complex gauge-invariant MMD target regularize the real encoded OLS coefficient law while retaining raw CMOS reconstruction?
PHYSICAL_LATENT: z∈C^100 raw aligned OLS coefficients
STATISTICAL_VARIABLE: S(z)=concat(Re(G(z)/σ), Im(G(z)/σ))∈R^200 with global-phase quotient
DATA_DIST: MNIST 55,000 TRAIN and 5,000 DEV in the consumer; protected/test are not training inputs
LATENT_EMPIRICAL_DIST: S(z) from real encoded samples at the M0 optical boundary
LATENT_PRIOR_DIST: native complex isotropic Gaussian z_j∼CN(0,σ²), σ=1e-3, ten target samples per batch
DISTRIBUTION_OBJECTIVE: full-complex Gaussian-kernel MMD V-statistic, bandwidth²=200; not q-only
SEMANTIC_OBJECTIVE: NONE; labels do not define a semantic generation loss here
IMAGE_OBJECTIVE: raw CMOS intensity MSE
OPTICAL_POWER_OBJECTIVE: NONE; configured optical weight γ=0
TRAINABLE_PARAMETERS: encoder.phase and generator.phase only; OLS and W_OLS remain active/frozen
SAMPLING_MECHANISM: deterministic seeded prior samples used as a training target only; no accepted novel-sample path
GENERATION_EXECUTED: NO_ACCEPTED_OPTICAL_GENERATION
SUPPORTED_CLAIM: V1 training consumer and full-complex MMD contract were executed within the bounded route
FORBIDDEN_CLAIM: MMD equals BOGT, prior samples equal optical generation, or semantic/physical readiness was established
CURRENT_NEXT_ACTION: Closed; no automatic Stage B, prior generation, or promotion
SOURCE_AUTHORITIES: lightgen_2d_semantic_complex_ols_training.py; lightgen_2d_semantic_generation_input_ols_alignment.py; CURRENT_STATUS.md; experiment_history.md
```

#### S1-V2：train-only normalized complex MMD

```text
ROUTE_ID: S1_COMPLEX_OLS_V2_NORMALIZED_MMD
ROLE: SUPPORT_BRANCH
STATUS: CLOSED_STAGE_A_NOT_ESTABLISHED
PARENT: M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION
SCIENTIFIC_QUESTION: Does train-only per-port scale normalization make a complex-OLS MMD warmup compatible with image reconstruction?
PHYSICAL_LATENT: z∈C^100, then z_tilde_j=z_j/sqrt(s_j+ε) using train-only s_j
STATISTICAL_VARIABLE: full-complex gauge-invariant normalized coefficient vector z_tilde, not raw q-only amplitude
DATA_DIST: MNIST TRAIN/DEV roles from the corrected V2 contract
LATENT_EMPIRICAL_DIST: normalized z_tilde from the real encoded OLS path after frozen TRAIN-only scales
LATENT_PRIOR_DIST: normalized complex CN(0,I), target of Stage B only
DISTRIBUTION_OBJECTIVE: Stage A has no MMD; Stage B would use full-complex normalized MMD after the Stage-A gate
SEMANTIC_OBJECTIVE: NONE; the route is distribution/reconstruction warmup, not semantic conditioning
IMAGE_OBJECTIVE: raw CMOS intensity MSE in Stage A and Stage B
OPTICAL_POWER_OBJECTIVE: Stage-B physical per-port scale anchor only; no optical-plane power-distribution loss
TRAINABLE_PARAMETERS: encoder.phase and generator.phase only; evaluator and OLS/W_OLS are frozen/active as declared
SAMPLING_MECHANISM: normalized prior target samples in Stage B only; no accepted image generation
GENERATION_EXECUTED: NO_ACCEPTED_OPTICAL_GENERATION
SUPPORTED_CLAIM: The corrected V2 contract and Stage-A outcome are recorded; Stage-A semantic warmup eligibility was not established
FORBIDDEN_CLAIM: V2 is a completed generator, normalized MMD is BOGT, or the Stage-B prior was validated
CURRENT_NEXT_ACTION: Closed; Stage B and generation stay closed after Stage-A failure
SOURCE_AUTHORITIES: lightgen_2d_semantic_complex_ols_training_v2.py; define/execute V2 contract scripts; CURRENT_STATUS.md; current_handoff.md
```

### 5.3 S2：semantic path diagnostic family V3–V11

```text
ROUTE_ID: S2_SEMANTIC_PATH_DIAGNOSTIC_V3_V11
ROLE: SUPPORT_BRANCH
STATUS: CLOSED_BOUNDED_SEMANTIC_DIAGNOSTICS
PARENT: S1_COMPLEX_OLS_V1_FULL_COMPLEX_MMD and S1_COMPLEX_OLS_V2_NORMALIZED_MMD
SCIENTIFIC_QUESTION: Which local semantic anchor, transfer, or low-rank remapping signal—if any—can move the existing OLS-aligned optical path without confusing semantic score with generative distribution?
PHYSICAL_LATENT: z∈C^100; V10/V10.1/V11 additionally use z_adapt=(I+UV^H)z with a rank-9 factorization
STATISTICAL_VARIABLE: SEMANTIC_FEATURE_DIST and geometric margins: prototype, evaluator feature, image-template, transfer, and OLS-direction quantities
DATA_DIST: MNIST TRAIN/DEV roles with train-only prototypes/templates and declared cyclic evaluation pairs
LATENT_EMPIRICAL_DIST: observed real encoded/adapted coefficient samples used for geometry diagnostics; no accepted fitted prior
LATENT_PRIOR_DIST: NONE
DISTRIBUTION_OBJECTIVE: NONE across V3–V11; these are semantic-path diagnostics, not latent probability matching
SEMANTIC_OBJECTIVE: V3 feature correction; V4 recoverability; V5/V6 prototype alignment; V7 transfer; V8/V9 anchor+transfer; V10/V10.1/V11 low-rank remapping
IMAGE_OBJECTIVE: raw CMOS L_image is a control/evaluation metric; V11 adds a differentiated λ_image L_image term, while earlier remap routes keep image terms evaluation-only
OPTICAL_POWER_OBJECTIVE: NONE
TRAINABLE_PARAMETERS: Variant-specific encoder.phase; V9 may add generator.phase; V10.1/V11 train rank-9 adapter factors only
SAMPLING_MECHANISM: deterministic class/cyclic pair selection and transfer z-shift; no random latent sampling
GENERATION_EXECUTED: NO_ACCEPTED_OPTICAL_GENERATION
SUPPORTED_CLAIM: Bounded semantic-path evidence and failure attribution can be compared by subroute; no subroute establishes generation readiness
FORBIDDEN_CLAIM: A semantic margin is a latent distribution, any V5–V11 result is BOGT, or cyclic transfer is novel CMOS generation
CURRENT_NEXT_ACTION: Closed for automatic escalation; any new objective, weight, architecture, or Stage-B action requires a new contract and explicit authorization
SOURCE_AUTHORITIES: define/execute V3–V11 contract and execution scripts; CURRENT_STATUS.md; current_handoff.md; experiment_history.md
```

S2 子路线的统一字段继承上卡；下表覆盖该子路线的 `STATUS`、主要 objective 和实际
边界。每个 ID 仍是独立注册对象，不能把父卡的 `CLOSED` 状态改写成 active。

| ROUTE_ID | STATUS | 覆盖的主要 objective / trainable surface | 当前判定 |
| --- | --- | --- | --- |
| `S2_V3_SEMANTIC_OBJECTIVE_CORRECTION` | `CLOSED_NOT_SUPPORTED` | frozen feature `L_feature` + `L_image`; encoder-only | provider-specific alignment 未支持；不开放下游。 |
| `S2_V4_FIXED_OLS_GENERATOR_RECOVERABILITY` | `CLOSED_NOT_SUPPORTED` | fixed-OLS generator recoverability; generator-only | recoverability 未支持；conditionality 后续诊断保持 unresolved。 |
| `S2_V5_DIRECT_COMPLEX_OLS_SEMANTIC_ALIGNMENT` | `CLOSED_NOT_SUPPORTED` | `L_image + λ_proto L_proto`; encoder-only | Smooth semantic gain 未支持；Stage-A firewall 未过。 |
| `S2_V6_PROTO_ONLY_SEMANTIC_WARMUP` | `CLOSED_NOT_SUPPORTED` | `L_proto` only; encoder-only; image loss evaluation-only | proto-only warmup 未支持。 |
| `S2_V7_TRANSFER_AWARE_SEMANTIC_WARMUP` | `CLOSED_NOT_SUPPORTED` | `L_transfer`; encoder-only; cyclic target `t=(y+1) mod 10` | transfer-aware warmup 未支持。 |
| `S2_V8_ABSOLUTE_ANCHOR_PLUS_TRANSFER` | `CLOSED_NOT_SUPPORTED` | `L_anchor + λ_transfer L_transfer`; encoder-only | anchor+transfer 未支持。 |
| `S2_V9_JOINT_SEMANTIC_REMAPPING` | `CLOSED_NOT_SUPPORTED` | V8 objective; encoder.phase + generator.phase | joint remapping 未支持；Stage-A 仍未达标。 |
| `S2_V10_LOW_RANK_REMAP_INVALID_SUPERSEDED` | `CLOSED_INVALID_SUPERSEDED` | `T_adapt=I+UV^H`; old step-0 identity harness | 旧 V10 是 invalid preflight；不与 V10.1 scientific 结果合并。 |
| `S2_V10_1_LOW_RANK_OLS_TO_GENERATOR_REMAP` | `CLOSED_BOUNDED_REMAP_ONLY` | rank-9 adapter factors; V9 semantic objective | bounded remap signal supported，但 image reconstruction恶化且 Stage-A firewall 未过。 |
| `S2_V11_LOW_RANK_SEMANTIC_RECONSTRUCTION_COMPATIBILITY` | `CLOSED_BOUNDED_COMPATIBILITY_ONLY` | V10.1 semantic objective + differentiated `λ_image L_image`; adapter-only | bounded compatibility Outer Gate supported，但不构成 Stage-A/generation 资格。 |

### 5.4 S3：N3 latent producer family

```text
ROUTE_ID: S3_N3_LATENT_DISTRIBUTION_PRODUCER_FAMILY
ROLE: SUPPORT_BRANCH
STATUS: FROZEN_COMPONENT_NO_ESCALATION
PARENT: M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION; N1/N2 are preparation/provenance predecessors
SCIENTIFIC_QUESTION: Can a declared producer match the observed OLS-amplitude geometry closely enough to be a reusable latent component without pretending to be a complete optical generator?
PHYSICAL_LATENT: observed complex OLS coefficients c∈C^100; primary producer coordinate q=abs(c/sqrt(s)); relative phase is a separate compatibility study
STATISTICAL_VARIABLE: q or R2_AMPLITUDE coordinates, empirical marginals, copula dependence, and separately tested relative phase
DATA_DIST: class-stratified N3 development source bank and its fit/heldout/sealed roles; not a population-data claim
LATENT_EMPIRICAL_DIST: empirical q/R2 amplitude marginals and dependence from the declared fit bank; observed complex coefficients remain the physical source
LATENT_PRIOR_DIST: selected development Gaussian-copula candidate and tested Gaussian/Student-t/mixture/phase candidates; none is authorized as M0 prior
DISTRIBUTION_OBJECTIVE: five-fold OLS compatibility, energy distance, covariance/norm/coordinate checks in the declared q-space or complex compatibility space
SEMANTIC_OBJECTIVE: NONE; labels stratify or balance the bank and do not create semantic image generation
IMAGE_OBJECTIVE: NONE
OPTICAL_POWER_OBJECTIVE: latent amplitude/power normalization and OLS-compatibility diagnostics only; no optical-plane power-loss training
TRAINABLE_PARAMETERS: fit-only producer state such as empirical marginals, copula/covariance, and model order; no full photonic generator parameters
SAMPLING_MECHANISM: producer-specific q sampling and candidate phase construction; materialized latent banks only
GENERATION_EXECUTED: NO_ACCEPTED_OPTICAL_GENERATION
SUPPORTED_CLAIM: A frozen development component/compatibility authority can be recorded; selected producer evidence is bounded to its declared source and consumer
FORBIDDEN_CLAIM: N3 is a complete z→W_OLS→generator→CMOS model, semantic generator, population prior, paper method, or hardware-readiness evidence
CURRENT_NEXT_ACTION: No escalation or M0 integration; a future consumer must define a new contract and reject incompatible phase/complex laws
SOURCE_AUTHORITIES: fit_n3_amplitude_producer_baseline.py; N3 model-order/producer scripts; CURRENT_STATUS.md; current_handoff.md; PROJECT_MEMORY.md
```

### 5.5 S4：architecture / nonlinear / physical-extension ablations

#### S4-A/B/C/D architecture × nonlinear family

```text
ROUTE_ID: S4_ABCD_ARCHITECTURE_NONLINEAR_ABLATION
ROLE: SUPPORT_BRANCH
STATUS: CLOSED_BOUNDED_ABLATION
PARENT: M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION and S2 semantic-path diagnostics
SCIENTIFIC_QUESTION: At the fixed OLS boundary, do 2L versus 4L generator depth or an ideal nonlinear facet change bounded semantic response under the same data and objective?
PHYSICAL_LATENT: existing z∈C^100 and optional rank-9 adapted coefficient; the A/B/C/D comparison does not create a new latent law
STATISTICAL_VARIABLE: SEMANTIC_FEATURE_DIST, raw CMOS recall, image loss, facet activation and optical-power retention by architecture/facet arm
DATA_DIST: fixed MNIST TRAIN/DEV and cyclic selections declared by each bounded ablation contract
LATENT_EMPIRICAL_DIST: fixed encoded/adapted coefficient observations; no fitted latent prior
LATENT_PRIOR_DIST: NONE
DISTRIBUTION_OBJECTIVE: NONE; this is an architecture/facet comparison, not a probability model
SEMANTIC_OBJECTIVE: anchor/transfer/Smooth comparative gate under the fixed optical interface
IMAGE_OBJECTIVE: raw CMOS intensity reconstruction MSE as objective or evaluation constraint per arm
OPTICAL_POWER_OBJECTIVE: raw mean power, facet activation, and retention diagnostics only; no power-distribution loss
TRAINABLE_PARAMETERS: arm-specific generator.phase and/or rank-9 adapter; encoder, OLS, W_OLS and frozen evaluator as declared
SAMPLING_MECHANISM: balanced multi-epoch schedule and deterministic cyclic transfer; no random latent sampling
GENERATION_EXECUTED: NO_ACCEPTED_OPTICAL_GENERATION
SUPPORTED_CLAIM: Bounded ideal-surrogate architecture/facet comparisons and the fixed generator-only continuation ceiling
FORBIDDEN_CLAIM: ideal ReLU is a material Te curve, depth comparison proves global optimality, or any arm establishes generation readiness
CURRENT_NEXT_ACTION: Closed; physical Te requires new material parameters and a separately authorized contract
SOURCE_AUTHORITIES: execute_lightgen_aligned_4l_relu_gain_v1.py; execute_lightgen_aligned_4l_linear_optimization_ceiling_v1.py; CURRENT_STATUS.md; current_handoff.md
```

The A/B/C/D identity is explicit: `A=2L Linear`, `B=2L ideal ReLU90`, `C=4L Linear`,
`D=4L ideal ReLU90`. N5, N6 and the continuation are registered S4 child IDs inheriting
the card above:

| ROUTE_ID | STATUS | 覆盖项 | 当前判定 |
| --- | --- | --- | --- |
| `S4_N5_TRUE_OPTICAL_DEPTH_COMPARISON` | `CLOSED_NOT_SUPPORTED` | C1 versus added 0.75 mm midpoint phase/propagation factorization | optimization horizon dominant；true-depth semantic gain 未支持。 |
| `S4_N6_NONLINEAR_FACET_ACTIVATION_CANARY` | `CLOSED_NOT_SUPPORTED` | fixed algorithmic nonlinear fiber-facet surrogate | nonlinear activation 未支持；不是 material validation。 |
| `S4_4L_LINEAR_OPTIMIZATION_CEILING_CONTINUATION` | `CLOSED_LINEAR_CONTINUATION_PLATEAU` | C2000→C3000 generator-only continuation | 当前固定优化路径进入平台；不等于全局最优。 |

#### Strict ReLU90

```text
ROUTE_ID: S4_STRICT_RELU90_TRAIN_CALIBRATED
ROLE: SUPPORT_BRANCH
STATUS: CLOSED_NOT_SUPPORTED
PARENT: S4_ABCD_ARCHITECTURE_NONLINEAR_ABLATION
SCIENTIFIC_QUESTION: Does a strictly TRAIN-calibrated ideal ReLU90 facet improve semantic response over nominal linear A under the fixed optical path?
PHYSICAL_LATENT: no new latent; local facet operation is I' = ReLU(I-τ) on optical intensity
STATISTICAL_VARIABLE: facet threshold τ, activation/suppression ratio, raw CMOS semantic recall and image loss
DATA_DIST: TRAIN calibration plus disjoint TRAIN validation and the declared DEV/cyclic evaluation roles
LATENT_EMPIRICAL_DIST: NONE; the route does not fit a latent probability law
LATENT_PRIOR_DIST: NONE; Strict ReLU90 is not a latent probability model
DISTRIBUTION_OBJECTIVE: NONE
SEMANTIC_OBJECTIVE: nominal-A versus strict TRAIN-calibrated ReLU90 semantic comparison
IMAGE_OBJECTIVE: raw CMOS intensity MSE and raw-validity constraint
OPTICAL_POWER_OBJECTIVE: facet activation and mean-power ratio diagnostics; no power-distribution loss
TRAINABLE_PARAMETERS: declared B-arm generator/adapter parameters only; threshold is TRAIN-calibrated, not a sampled latent
SAMPLING_MECHANISM: balanced/cyclic image-label schedule; no latent sampling
GENERATION_EXECUTED: NO_ACCEPTED_OPTICAL_GENERATION
SUPPORTED_CLAIM: The strict ideal algorithmic ReLU90 comparison was bounded and its semantic gain was not supported
FORBIDDEN_CLAIM: ReLU90 defines a latent prior, models a physical Te response, or proves optical generation readiness
CURRENT_NEXT_ACTION: Closed; no retuning or material extrapolation without a new contract
SOURCE_AUTHORITIES: execute_lightgen_2d_strict_train_calibrated_relu90_gain_v2.py; CURRENT_STATUS.md; current_handoff.md
```

### 5.6 B0：formal `label + 8D` compatibility baseline

```text
ROUTE_ID: B0_FORMAL_LABEL8D_IDENTITY_OLS_COMPATIBILITY
ROLE: BASELINE
STATUS: FROZEN_COMPATIBILITY_BASELINE
PARENT: v2 teacher-free formal runtime
SCIENTIFIC_QUESTION: Does the preserved formal CLI/runtime contract remain structurally compatible with label-conditioned 8D amplitude-SLM control and identity OLS?
PHYSICAL_LATENT: electronic label y∈{0,…,9} plus code c∈R^8; identity OLS is a complex-field transport boundary, not a sampled latent
STATISTICAL_VARIABLE: label and 8D code; no OLS-native q/phase probability law is defined
DATA_DIST: formal MNIST split roles for training/evaluation only; no new producer distribution
LATENT_EMPIRICAL_DIST: NONE defined for this compatibility record
LATENT_PRIOR_DIST: NONE in this registry; any external code sampler is contract-specific and not accepted here
DISTRIBUTION_OBJECTIVE: NONE in the structural compatibility baseline
SEMANTIC_OBJECTIVE: label-conditioned carrier compatibility only; no current semantic-generation acceptance
IMAGE_OBJECTIVE: raw CMOS intensity path is defined by the formal runtime; current acceptance is not established
OPTICAL_POWER_OBJECTIVE: fixed exposure, passivity and calibration diagnostics; no fitted optical-power law
TRAINABLE_PARAMETERS: formal electronic generator and phase-only metasurfaces exist in the runtime; no active authorized update is recorded by this route
SAMPLING_MECHANISM: generic formal sample command may provide label/code; inference remains single-exposure and current readiness is not established
GENERATION_EXECUTED: NO_ACCEPTED_BASELINE_GENERATION
SUPPORTED_CLAIM: Formal v2 architecture/CLI compatibility and the label+8D → amplitude SLM → identity OLS chain
FORBIDDEN_CLAIM: B0 is the current M0 input, a LightGen paper reproduction, a learned OLS-native manifold, or accepted hardware generation
CURRENT_NEXT_ACTION: Preserve as baseline; do not automatically resume formal training or redirect M0 to B0
SOURCE_AUTHORITIES: v2_runtime.md; conditional runtime; identity_generator.py; formal_identity_logit_amplitude_finite_crop_identity.yaml; PROJECT_MEMORY.md
```

### 5.7 H1：historical scaled-Chen BOGT / reconstruction route

```text
ROUTE_ID: H1_SCALED_CHEN_BOGT_RECONSTRUCTION_REPAIR
ROLE: HISTORICAL_CLOSED
STATUS: HISTORICAL_CLOSED
PARENT: historical SMF/Chen reconstruction branch
SCIENTIFIC_QUESTION: Can a scaled Chen-reference SMF geometry combine camera reconstruction, aggregate latent-moment regularization and optical-power terms in a historical diagnostic route?
PHYSICAL_LATENT: scaled LP01 100-port OLS coefficients; route losses use normalized port power and aggregate moments
STATISTICAL_VARIABLE: normalized port powers, aggregate moments, and historical BOGT/KL-surrogate coordinates
DATA_DIST: historical reconstruction-task data and dated run roles; exact split is owned by the cited historical reports
LATENT_EMPIRICAL_DIST: observed scaled-Chen OLS port-power observations when calibration is supplied
LATENT_PRIOR_DIST: historical truncated-normal port-power sampler exposed by the route code; not a current M0 prior
DISTRIBUTION_OBJECTIVE: paper-inspired aggregate bogt_kl_surrogate / moment constraint; exact paper implementation and unpublished details remain unknown
SEMANTIC_OBJECTIVE: historical reconstruction/readout diagnostics; no accepted semantic generation contract
IMAGE_OBJECTIVE: camera/raw CMOS reconstruction MSE
OPTICAL_POWER_OBJECTIVE: historical op=lop_from_powers and operating-window power variants; route-specific diagnostics, not current power evidence
TRAINABLE_PARAMETERS: historical encoder.phase and decoder.phase; scaled OLS remains fixed
SAMPLING_MECHANISM: historical sample_prior_coefficients followed by decoder/CMOS re-emission capability
GENERATION_EXECUTED: NO_ACCEPTED_CURRENT_GENERATION_EVIDENCE
SUPPORTED_CLAIM: Historical code and bounded reconstruction/objective hypotheses are preserved for provenance
FORBIDDEN_CLAIM: Exact paper BOGT reproduction, current LightGen implementation, accepted prior generation, semantic novelty, or hardware readiness
CURRENT_NEXT_ACTION: Archive only; no historical next step is active in the current tree
SOURCE_AUTHORITIES: optical_manifold_smf_bogt.py; lightgen_scaled_chen_bogt_v1.md; lightgen_optical_manifold_v1.md; experiment_history.md; current_handoff.md
```

历史 Scaled-Chen 分支下的旧 `S1–S3 controller`、rank-16 electronic readout、旧 closed
reconstruction/readout 只作为 `H1` 的历史 sibling；它们不获得新的 active `ROUTE_ID`，也
不改变 M0 的唯一性。

### 5.8 H2：v1 LDM teacher historical runtime

```text
ROUTE_ID: H2_V1_LDM_TEACHER_REPRODUCTION
ROLE: HISTORICAL_CLOSED
STATUS: HISTORICAL_REPRODUCTION_RUNTIME
PARENT: v1 D2NN single-exposure distillation line
SCIENTIFIC_QUESTION: Can the historical teacher-first-stage/diffusion/target-export chain reproduce the project’s v1 distillation inputs?
PHYSICAL_LATENT: teacher first-stage VAE latent and diffusion latent/target payload; this is separate from M0 complex OLS z
STATISTICAL_VARIABLE: teacher latent/target-image payload distributions and optional label/readability audit features
DATA_DIST: historical v1 MNIST teacher-training, export and distillation split roles
LATENT_EMPIRICAL_DIST: first-stage VAE latent samples and exported teacher target payloads in the v1 chain
LATENT_PRIOR_DIST: diffusion noise/latent law matched to the paired first-stage decoder; historical contract only
DISTRIBUTION_OBJECTIVE: historical VAE/diffusion and distillation losses; not an OLS-native MMD or Chen-inspired BOGT objective
SEMANTIC_OBJECTIVE: optional historical label/readability/distillation terms; no current LightGen semantic claim
IMAGE_OBJECTIVE: teacher target reconstruction and one-shot student camera/image losses in the historical v1 workflow
OPTICAL_POWER_OBJECTIVE: NONE as a current LightGen acceptance objective
TRAINABLE_PARAMETERS: historical teacher first-stage/diffusion and v1 student parameters by stage
SAMPLING_MECHANISM: multi-step software diffusion sampling and teacher export; final hardware contract must not depend on this path
GENERATION_EXECUTED: HISTORICAL_SOFTWARE_ONLY_NOT_CURRENT_CMOS_ACCEPTANCE
SUPPORTED_CLAIM: v1 teacher/diffusion reproducibility and target-export provenance
FORBIDDEN_CLAIM: v1 teacher is the v2 teacher-free runtime, a LightGen OLS-native generator, or hardware/single-exposure readiness
CURRENT_NEXT_ACTION: Preserve for historical reproduction only; do not import it into v2 or M0
SOURCE_AUTHORITIES: v1_single_shot_distillation.md; train.py; src/ldm_project/v1/teacher; src/ldm_project/v1/student; experiment_history.md
```

## 6. 训练目标统一矩阵

矩阵中的 `empirical distribution` 与 `target prior` 均按第 2 节的限定术语解释；表内不
使用未声明轴的裸“distribution”。`NO` 表示没有已接受的光学 CMOS generation，
即使路线内部曾采样 target、latent bank 或运行 evaluator。

| Route | latent | empirical distribution | target prior | distribution loss | semantic loss | image loss | power loss | sampling | generation status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | planned `z∈C^100` | `LATENT_EMPIRICAL_DIST(z)`: UNKNOWN | `LATENT_PRIOR_DIST(z)`: UNKNOWN | planned Chen-inspired aggregate, TBD | planned raw-CMOS semantic, TBD | candidate raw-CMOS MSE, TBD | planned optical-power term, TBD | TBD external z producer | NO |
| `S1_COMPLEX_OLS_V1_FULL_COMPLEX_MMD` | raw `z∈C^100`, `S(z)∈R^200` | `LATENT_EMPIRICAL_DIST(S(z))` | `LATENT_PRIOR_DIST(z)=CN(0,σ²)` | full-complex gauge-invariant MMD V-stat | NONE | raw CMOS intensity MSE | NONE | target prior only | NO |
| `S1_COMPLEX_OLS_V2_NORMALIZED_MMD` | normalized `z_tilde∈C^100` | `LATENT_EMPIRICAL_DIST(z_tilde)` after train-only scale | normalized `CN(0,I)` in Stage B | Stage-A NONE; Stage-B normalized full-complex MMD | NONE | raw CMOS intensity MSE | scale anchor, not optical-power law | Stage-B target prior only | NO |
| `S2_V3_SEMANTIC_OBJECTIVE_CORRECTION` | OLS `z∈C^100` | observed coefficient geometry | NONE | NONE | feature alignment correction | raw CMOS metric | NONE | deterministic paired batches | NO |
| `S2_V4_FIXED_OLS_GENERATOR_RECOVERABILITY` | OLS `z∈C^100` | observed fixed-OLS geometry | NONE | NONE | recoverability/conditionality diagnosis | raw CMOS metric | NONE | deterministic paired batches | NO |
| `S2_V5_DIRECT_COMPLEX_OLS_SEMANTIC_ALIGNMENT` | OLS `z∈C^100` | observed encoded z and prototypes | NONE | NONE | `L_proto` plus image term | raw CMOS MSE | NONE | class-balanced prototypes | NO |
| `S2_V6_PROTO_ONLY_SEMANTIC_WARMUP` | OLS `z∈C^100` | observed z/prototype geometry | NONE | NONE | `L_proto` only | evaluation-only raw CMOS MSE | NONE | deterministic cyclic target | NO |
| `S2_V7_TRANSFER_AWARE_SEMANTIC_WARMUP` | OLS `z∈C^100` and shifted z | observed transfer geometry | NONE | NONE | `L_transfer` | evaluation-only raw CMOS MSE | NONE | `t=(y+1) mod 10` | NO |
| `S2_V8_ABSOLUTE_ANCHOR_PLUS_TRANSFER` | OLS z and image-template anchor | observed semantic/image geometry | NONE | NONE | `L_anchor+λ_transfer L_transfer` | evaluation-only raw CMOS MSE | NONE | deterministic cyclic transfer | NO |
| `S2_V9_JOINT_SEMANTIC_REMAPPING` | OLS z and joint remap | observed semantic/image geometry | NONE | NONE | V8 objective | evaluation-only raw CMOS MSE | NONE | deterministic cyclic transfer | NO |
| `S2_V10_LOW_RANK_REMAP_INVALID_SUPERSEDED` | adapted OLS z, invalid old adapter | observed adapter identity probe | NONE | NONE | V9 objective not reached | pre-scientific only | NONE | none; invalid preflight | NO |
| `S2_V10_1_LOW_RANK_OLS_TO_GENERATOR_REMAP` | `z_adapt=(I+UV^H)z` | adapted coefficient/semantic geometry | NONE | NONE | anchor+transfer | evaluation metric; worsened reconstruction | NONE | deterministic remap/cyclic transfer | NO |
| `S2_V11_LOW_RANK_SEMANTIC_RECONSTRUCTION_COMPATIBILITY` | adapted z plus raw CMOS | adapted coefficient and semantic geometry | NONE | NONE | anchor+transfer | differentiated `λ_image L_image` | NONE | deterministic remap/cyclic transfer | NO |
| `S3_N3_LATENT_DISTRIBUTION_PRODUCER_FAMILY` | `q=abs(c/sqrt(s))`, c∈C^100 | `LATENT_EMPIRICAL_DIST(q)` and dependence bank | candidate copula/model-specific q law | five-fold q/OLS compatibility and energy distance | NONE | NONE | latent power normalization diagnostics | q/phase candidate bank | NO |
| `S4_ABCD_ARCHITECTURE_NONLINEAR_ABLATION` | fixed OLS z; no new law | observed architecture/facet response | NONE | NONE | anchor/transfer/Smooth comparison | raw CMOS MSE | activation/retention diagnostic | balanced/cyclic schedule | NO |
| `S4_N5_TRUE_OPTICAL_DEPTH_COMPARISON` | fixed OLS z | observed true-depth response | NONE | NONE | bounded semantic comparison | raw CMOS MSE | raw power diagnostic | balanced/cyclic schedule | NO |
| `S4_N6_NONLINEAR_FACET_ACTIVATION_CANARY` | fixed OLS z | observed nonlinear-facet response | NONE | NONE | bounded semantic comparison | raw CMOS MSE | activation/power diagnostic | balanced/cyclic schedule | NO |
| `S4_STRICT_RELU90_TRAIN_CALIBRATED` | fixed OLS z; facet `ReLU(I-τ)` | NONE; no latent law | NONE | NONE | strict ReLU90 versus nominal A | raw CMOS MSE | activation/mean-power diagnostic | balanced/cyclic schedule | NO |
| `S4_4L_LINEAR_OPTIMIZATION_CEILING_CONTINUATION` | fixed OLS z | observed C2000/C3000 response | NONE | NONE | anchor+transfer | raw CMOS MSE | raw power diagnostic | resumed balanced schedule | NO |
| `B0_FORMAL_LABEL8D_IDENTITY_OLS_COMPATIBILITY` | label + `c∈R^8` | NONE defined | NONE defined | NONE | label-conditioned compatibility | formal raw CMOS path | fixed exposure/passivity diagnostic | external formal code input | NO |
| `H1_SCALED_CHEN_BOGT_RECONSTRUCTION_REPAIR` | scaled-Chen port powers/moments | observed port-power bank | historical truncated-normal port-power law | aggregate KL/BOGT surrogate | reconstruction/readout diagnostic | camera MSE | `lop_from_powers` historical term | historical prior re-emission | NO current acceptance |
| `H2_V1_LDM_TEACHER_REPRODUCTION` | VAE/diffusion latent payload | historical teacher target bank | paired diffusion latent/noise law | historical teacher/distillation losses | historical label/readability terms | teacher/student image losses | NONE current | multi-step software diffusion | historical software only |

## 7. 静态一致性复核与未决项

### `LIGHTGEN_ROUTE_AND_DISTRIBUTION_REGISTRY_V1_CONSISTENT`

**PASS（docs-only static review）**。复核项如下：

- 所有登记 `ROUTE_ID` 唯一；S2、S4 的子路线使用父卡继承，不重复占用另一个主线
  身份。
- mainline role 只有 M0；其余路线分别为 support、baseline 或 historical closed。
- 每张路线卡只有一个 `STATUS`；`CLOSED_*`、`FROZEN_*` 和 `HISTORICAL_*` 路线没有
  active next action。
- S1-V1/V2 明确是 MMD，不写成 BOGT；H1 明确是 paper-inspired historical surrogate，
  不写成 exact paper method。
- S2 的 semantic/prototype/transfer/margin 只登记为 `SEMANTIC_FEATURE_DIST` 或几何
  diagnostic；没有被写成 `LATENT_PRIOR_DIST` 或 generation。
- S3-N3 只登记 producer component 与 latent-bank sampling；没有被写成完整 generator、
  semantic generator 或 optical generation。
- S4-Strict 明确是 optical-facet intensity surrogate，`LATENT_PRIOR_DIST=NONE`，没有被
  写成 latent probability model；S4 也没有被写成 physical Te validation。
- B0 与 H2 保留各自架构/历史身份，未被并入 M0；当前 registry 不改变任何训练、Stage B、
  generation 或 promotion gate。

### `UNKNOWN / NEEDS_CLASSIFICATION`

- M0 的确切 BOGT 统计轴、target law、权重、trainable groups、semantic/image/power
  联合合同和 external-z sampling mechanism 尚未定义。
- Chen/LightGen 未披露或项目未验证的 normalization、phase write、port order、modal
  normalization、完整 relay/aperture/generator 细节仍是 `UNKNOWN`，不能由 H1 或 N3 补齐。
- N3 是否可作为 M0 的一个输入 producer 尚未有被命名的 consumer contract；当前不自动接入。
- S4 的 ideal ReLU、true optical depth、future Te/material curve 与物理器件验证仍未分类为
  formal evidence；下一步若发生必须另起 contract。
- 任何路线的 generation readiness、novelty/diversity 和 hardware readiness 均没有因本
  注册表创建而改变。

### `AUTHORITY_CONFLICTS`

本次核对未发现需要仲裁的 authority conflict。`PROJECT_MEMORY` 保留稳定路线身份，
`CURRENT_STATUS`/`current_handoff` 保留 2026-09-16 之后的数值和停止边界，历史文件继续
保留日期化事实；三者分工不同，不把旧文档中的历史“下一步”解释为当前授权。

## 8. 本次整理结果

```text
RESULT: CREATED_CANONICAL_LIGHTGEN_ROUTE_AND_DISTRIBUTION_REGISTRY
GIT_BRANCH: research/lightgen-ols-native-semantic-generation-v1
GIT_HEAD: a9cd1749adfa690e282fd52a6425564fc2cf102c
GIT_DIRTY: YES (pre-existing user changes preserved; current docs delta added)
OUTER_GATE: LIGHTGEN_ROUTE_AND_DISTRIBUTION_REGISTRY_V1_CONSISTENT=PASS
NEXT: no scientific execution; if continuing, define one bounded M0 contract before any training
```
