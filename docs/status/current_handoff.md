# 当前研究交接（2026-09-10）

## 本次目标增量

用户明确要求：输入与 OLS 对齐陈一同等人的 LightGen，衍射层架构可以不同；尽快得到语义生成原型。
唯一活动研究主线及路线地位见 [项目记忆](../PROJECT_MEMORY.md#唯一活动研究主线lightgen-输入与-ols-对齐的语义生成原型)，
已完成科学结果统一查 [当前状态](../CURRENT_STATUS.md)。不把正式 `label + 8D` 基线或历史 reconstruction
closeout 当成当前研究目标；也不把 N3 分布支持当成 LightGen 输入/OLS 对齐完成。

## 当前交接结果：LightGen 2D real-path control diagnosis 已完成

本次 joint-compatibility Gate 已按授权完成并关闭独立 phase candidate；随后 `Q_CONDITIONED_NEAREST_PHASE_DONOR_V1`
按合同执行为 `0/5`（control=`5/5`）。higher-expressivity candidate
`Q_CONDITIONED_COMPLEX_PHASOR_LINEAR_RESIDUAL_BOOTSTRAP_V1` 也已按 READY contract 完成一次且仅一次，结果为
`0/5`，classification=`N3_COMPLEX_OLS_Q_CONDITIONED_COMPLEX_PHASOR_LINEAR_RESIDUAL_NOT_SUPPORTED`。
随后 bounded architectural decision 已完成并采用 `LIGHTGEN_INPUT_OLS_ALIGNMENT_FIRST`；definition-only
契约已 READY，aligned input/complex-OLS runtime、training consumer/preflight 与正式 100-step CUDA training
均已完成实现和验证。用户随后对唯一 post-training evaluation scope 已独立明确授权并已被消费；该 evaluation
按 READY contract 完成一次且仅一次，结果为 `VALID_COMPLETED`，但 Stage 1 full protected real-path control 未建立，
因此流程按合同停止，未进入 latent compatibility、generation、semantic distribution、diversity 或 novelty。
随后用户独立明确授权一次 bounded real-path control diagnosis；该 diagnosis 已 `VALID_COMPLETED`，primary
diagnosis=`REAL_PATH_FAILURE_ENCODER_OLS_SEMANTIC_COLLAPSE_SUPPORTED`。
不得把项目自建假设表述成论文事实，
也不得未经新的明确授权就从文档进入训练或新样本生成。论文输入、OLS 物理算子与模式耦合的差异审查已被
契约显式分为 CHEN_FACT、CHEN_UNRESOLVED 和 PROJECT_DECISION。

## 最新 V2 TRAIN source authority resolution：合同转录错误确认（2026-09-10）

对 `RESOLVE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_V2_TRAINING_SOURCE_AUTHORITY_V1` 完成一次只读 provenance
resolution。结果为 `VALID_COMPLETED`，classification=`TRAINING_SOURCE_AUTHORITY_CONTRACT_TRANSCRIPTION_ERROR_CONFIRMED`。
correction authority 为
[`lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction.v1.json`](../../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction_v1_attempt_01/lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction.v1.json)，
SHA-256=`fdf03a0a8568db947011549e7869ff11ab972e19c1e01fbc3e7d2e9d850dbeb0`；corrected contract 为
[`lightgen_2d_semantic_complex_ols_training_contract.v2_corrected.v1.json`](../../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction_v1_attempt_01/lightgen_2d_semantic_complex_ols_training_contract.v2_corrected.v1.json)，
SHA-256=`aaa3cc409a9a4775be4bf48ef566fce8615f75b72277ca79194d36bd1f56c9cf`。

原 V2 contract SHA-256=`b85fce30b70643638c1ea276d78dc58174c90980e420414cafc3f1947155de53` 中的
`data_contract.source_artifacts.training.sha256` 是 63 位 malformed transcription
`8151bb33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd`；当前未修改的
`data/MNIST/processed/training.pt` SHA-256=`8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd`，
大小=`47,521,438` bytes、schema=`60000×28×28 uint8 + 60000 int64`。N3 development source manifest、V1
training contract/consumer/preflight/training report/lineage 及 post-training contract 的 exact-source
claims 全部指向该 canonical SHA；无 competing valid digest，故确认是合同转录错误而非 source file drift。

corrected contract 仅修正 source SHA 并添加 correction lineage，semantic normal form 与原合同相同
（SHA=`cb90ac1bb33d56b155abf1cb2b01776983af356babbc45ccb8dded9ebd203203`）；原合同保留，source 保留，
科学语义未变。当前 V2 consumer 仍绑定原 contract，`consumer_rebind_required=true`。本次没有 CUDA、optimizer
steps、scientific training、generation 或 protected/test/sealed read，治理标志保持关闭。

下一动作固定为：先将 V2 consumer/preflight rebind 到 corrected contract，再以新 attempt 重跑
`PREFLIGHT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_V2`；不得由 correction 自动授权训练。

## 最新 V2 long-training preflight：corrected authority 可执行（2026-09-11）

上述 rebind 已完成：active V2 consumer/preflight 现在绑定 corrected contract
SHA-256=`aaa3cc409a9a4775be4bf48ef566fce8615f75b72277ca79194d36bd1f56c9cf`，correction authority
SHA-256=`fdf03a0a8568db947011549e7869ff11ab972e19c1e01fbc3e7d2e9d850dbeb0`，canonical
`training.pt` SHA-256=`8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd`。原 malformed contract
SHA-256=`b85fce30b70643638c1ea276d78dc58174c90980e420414cafc3f1947155de53` 未覆盖，继续作为 historical-only authority；
corrected/original scientific semantic normal form 相同，SHA-256=`cb90ac1bb33d56b155abf1cb2b01776983af356babbc45ccb8dded9ebd203203`。

按完全相同的冻结协议重跑 `PREFLIGHT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_V2`，attempt-07 结果为
`valid_completed/AUTHORIZABLE`，classification=`LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_V2_AUTHORIZABLE`。
input、report、lineage 分别为：

- [`preflight input`](../../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_long_training_preflight_v2_attempt_07/lightgen_2d_semantic_complex_ols_long_training_preflight_v2_input.v1.json)，SHA-256=`2e090926b057f70ed9cacc63473d16786336aa39b9371da2dcfdbccfead94b1d`；
- [`preflight report`](../../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_long_training_preflight_v2_attempt_07/lightgen_2d_semantic_complex_ols_long_training_preflight_v2_report.v1.json)，SHA-256=`b7bc2482d55af70ad8fc7a1b204c636425e9bca64448feda9e11e0111b5d64a4`；
- [`preflight lineage`](../../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_long_training_preflight_v2_attempt_07/lightgen_2d_semantic_complex_ols_long_training_preflight_v2_lineage.v1.json)，SHA-256=`30913cd07ad318fbd94d2e5a4a346da6ca8390087e47a0b145d2c08b54b2d3bd`。

目标 CUDA 实际为 RTX 3060 Laptop GPU、compute capability=`8.6`、PyTorch=`2.5.1`、CUDA build=`12.1`、VRAM=`6,441,926,656`
bytes。Stage A/B 各完成 3-step、grid=`2048`、batch=`10`、Adam lr=`1e-3` engineering smoke；Stage A 的 MMD 关闭和
modal projection → `T=I_100` → `W_OLS` → generator 路径通过，Stage B 的 normalized full-complex MMD/raw-z scale-anchor
路由、frozen scale 与 trainable firewall 通过。55,000 TRAIN-only scale traversal 通过（649.11785 s，约 84.73 rows/s，
float64，epsilon=`1e-12`）；preflight scale 与 preflight checkpoint 均被 formal loader 拒绝。A/B checkpoint/resume 的
loss 与 parameter 最大差异均为 `0.0`，optimizer 等价且 scale identity 保持；peak reserved VRAM=`4,592,762,880` bytes，
minimum free=`557,125,632` bytes，storage feasible。

本次 data firewall 为 TRAIN=`55,000`、DEV semantic smoke=`1,000`、calibration/protected/test/sealed=`0`，没有 300+300
scientific training、generation、heldout evaluation 或 promotion。`training_authorized=false`、`formal_promotion=false`、
`generation_readiness=not_ready` 不变。attempt-02 的 authority blocker 仍保持 historical/superseded-by-authority-correction，
attempt-03 至 attempt-06 的 harness 修复 artifacts 也未覆盖。下一动作只能是独立明确授权后的
`EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V2`，不得自动启动。

## 最新实现结果：LightGen 2D semantic complex-OLS training consumer V2（2026-09-10）

已按 hash-bound V2 contract 完成一次且仅一次 implementation consumer，结果为
`valid_completed/IMPLEMENTED`，classification=`LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONSUMER_V2_IMPLEMENTED`。
V2 contract SHA-256=`b85fce30b70643638c1ea276d78dc58174c90980e420414cafc3f1947155de53`；consumer SHA-256=
`d8b80121faf9da618a1710644b28fbf22755615aafba4d92cb0facb777f5fe97`。
实现报告位于
[`lightgen_2d_semantic_complex_ols_training_consumer_v2_implementation_report.v1.json`](../../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_training_consumer_v2_attempt_02/lightgen_2d_semantic_complex_ols_training_consumer_v2_implementation_report.v1.json)，
SHA-256=`8c6a07e9abf302a2060b4e65b91b65d5c514cdf1257576eba5936f734e28ebbe`；preflight input SHA-256=
`93773f3f88daf8be73ba576e6264acfd093e02eb77af877818e6798d0b6a588c`。

consumer 已实现完整的两阶段 state machine、Stage-A/B loss boundary、train-only physical scale、normalized
full-complex prior/MMD、optimizer reset、checkpoint identity/resume 与 modal projection/`W_OLS` autograd firewall。
工程 smoke 仅使用 synthetic/CPU 128-grid，Stage A/B 各 1 update；scientific steps、real/protected/test/sealed rows
均为 0。此次没有 CUDA training、heldout evaluation、generation acceptance 或 governance promotion。
`training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready` 保持；原定的
`PREFLIGHT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_V2` 已在新 attempt-02 因 TRAIN source authority
mismatch 有效阻断，该 mismatch 后续已被独立 correction 确认为合同转录错误；后续仍需 consumer rebind、重新
完成 preflight，并取得独立明确训练授权。

## 最新 V2 long-training preflight：authority gate 阻断

已按上述 preflight 输入创建新 attempt-02 并完成 authority 预检；结果为 `valid_failed/BLOCKED`，classification=
`BLOCKED_TRAINING_SOURCE_AUTHORITY_MISMATCH`。报告 SHA-256=`2fbea9060fc96a2634201366dc2cb5bb549491581cc4fca3aa518a033750f7b3`，
lineage SHA-256=`1f9abc34454ef26734081c2b5c86352d624970b359558ccc8542ce2f71ddae74`。

冻结 V2 contract SHA-256=`b85fce30b70643638c1ea276d78dc58174c90980e420414cafc3f1947155de53` 所绑定的
`training.pt` SHA-256=`8151bb33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd`，与当前本地文件实际
SHA-256=`8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd` 不一致。该冲突不能通过改写
冻结合同或静默接受本地文件解决，因此 preflight 在训练张量加载前 fail-closed 停止；没有进入 CUDA、DEV
semantic smoke、scale extraction、checkpoint/resume 或 300+300 training。protected/test/sealed 均未读取，
三个治理标志保持关闭。source authority 已由独立 correction artifact 解析完成；下一动作是先完成 consumer
rebind，再用新 attempt 重走完整 preflight；不得绕过该 gate。

| 必须明确的接口 | 已知差异/问题 | 交接需要给出的结果 |
| --- | --- | --- |
| 图像 → SLM | 论文确认 amplitude SLM，但精确灰度映射未核实；项目显式冻结 sqrt 作为 PROJECT_DECISION | 下一实现必须输出 X、I_slm、A_slm、E_slm、PlaneSpec 与 provenance |
| OLS → generator 输入 | 论文确认 generator-only 方向；项目必须使用复数 100-port 接口 | 实现唯一 `W_OLS(z)=sum_j(Tz)_j v_j`，并让 real/generated 两路共用 |
| OLS 物理算子 | 研究线是缩尺度 LP01 投影/重发，正式 v2 是 field identity | 使用 10×10/100-port complex modal boundary；T=I_100 仅为系数空间 PROJECT_DECISION |
| 分布训练与采样 | 历史 BOGT/producer 细节不能替代完整 complex-OLS manifold | V2 consumer=`IMPLEMENTED` 且已 rebind corrected authority；V2 long-training preflight=`AUTHORIZABLE`；V2 formal training 尚未授权；旧 V1 training/evaluation 与 diagnosis 仅为历史证据；generation 未执行 |
| 当前 alignment contract | `READY`；z∈C^100、强制 projection/re-emission、row-major port order、same writer 已冻结 | runtime、consumer、training、evaluation contract、一次性 evaluation 与 bounded diagnosis 已完成；generation acceptance、formal promotion、sealed reuse 仍关闭 |

## 当前 definition 结果：LightGen 2D complex-OLS post-training evaluation（2026-09-10）

契约为
[`lightgen_2d_semantic_complex_ols_post_training_evaluation_contract.v1.json`](../../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_post_training_evaluation_contract_v1_attempt_01/lightgen_2d_semantic_complex_ols_post_training_evaluation_contract.v1.json)，SHA-256=
`b750bafd9c3866707a7d492c55c12ee08ab4775da9f6878805317feccb937a1d`，status/result=`valid_completed/READY`，
classification=`DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_POST_TRAINING_EVALUATION_CONTRACT_V1_READY`。

它把后续一次性执行固定为唯一 step-100 checkpoint、同一 real/generated `W_OLS` 与 generator、accepted independent
`FrozenMNISTEvaluator v1`、4000-row development calibration reserve、10000-row protected test role 及固定 1000-row
protected panel。raw CMOS 只允许 clamp `[0,1]` + adaptive average pool `28×28`；该视图是 PROJECT_DECISION，旧的
0.9712 output-domain evaluator 因低于 0.985 gate 被拒绝。latent compatibility 使用训练一致的 full-complex gauge/MMD，
semantic primary 使用唯一 calibrated feature-MMD，diversity/novelty 为固定 one-sided real-null HARD tests；B=`999`、alpha=`0.01`。

本次 definition 未读取 protected values，未执行 MMD、CUDA、training、generation 或 heldout evaluation；sealed 仍不可用且不重用。
`training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready` 保持。下一步仅是独立明确授权后的
`EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_POST_TRAINING_EVALUATION_V1`，不得自动开始。

## 当前 post-training evaluation 结果：LightGen 2D semantic complex-OLS（2026-09-10）

按上述 READY contract，在用户独立明确授权后完成一次且仅一次
`EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_POST_TRAINING_EVALUATION_V1`。结果为
`VALID_COMPLETED`，classification=`LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_POST_TRAINING_CONTROL_NOT_ESTABLISHED`。

固定 step-100 checkpoint SHA-256=`6cfcc82831b575d83c097648a78b6507b32900e578d00c446d76ccd29130e149`；
evaluation report 为
[`lightgen_2d_semantic_complex_ols_post_training_evaluation_report.v1.json`](../../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_post_training_evaluation_v1_attempt_01/lightgen_2d_semantic_complex_ols_post_training_evaluation_report.v1.json)，
SHA-256=`4048de57a35a1899240a231cfbeec70ed5950275b2014fbcb1dfc96c664c4996`；lineage 为
[`lightgen_2d_semantic_complex_ols_post_training_evaluation_lineage.v1.json`](../../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_post_training_evaluation_v1_attempt_01/lightgen_2d_semantic_complex_ols_post_training_evaluation_lineage.v1.json)，
SHA-256=`9f1a89bafa8f57f2d6cd4242de2153a2cd5d4c02ac2d2ad82a652680a204f19a`；control artifact 为
[`lightgen_2d_semantic_complex_ols_post_training_evaluation_control.v1.pt`](../../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_post_training_evaluation_v1_attempt_01/lightgen_2d_semantic_complex_ols_post_training_evaluation_control.v1.pt)，
SHA-256=`c5b2cf0bd2d98b5eceb4a53881312c1397bda91e68a87def788364d72244a7a3`。

Stage 1 使用完整 10000-row protected real path；clean calibration macro recall=`0.987`，
冻结 threshold=`0.937`，protected real-path macro recall=`0.16177222311773765`，故 control=`FAIL`。
latent compatibility、direct generation、semantic distribution、diversity 和 novelty 均为
`NOT_EXECUTED_CONTROL_FAILED`；没有 training、optimizer/backward、checkpoint search、threshold tuning、
generation acceptance 或 sealed load/reuse。`training_authorized=false`、`formal_promotion=false`、
`generation_readiness=not_ready` 保持。下一步只能是围绕该 control failure 的 bounded diagnosis；不得自动重训、重跑或打开 sealed。

## 当前 bounded diagnosis 结果：LightGen 2D real-path control（2026-09-10）

按 post-training control failure 后的独立授权完成一次且仅一次
`DIAGNOSE_LIGHTGEN_2D_SEMANTIC_REAL_PATH_CONTROL_FAILURE_V1`。结果为 `VALID_COMPLETED`，primary diagnosis 为
`REAL_PATH_FAILURE_ENCODER_OLS_SEMANTIC_COLLAPSE_SUPPORTED`。

report 为
[`lightgen_2d_semantic_real_path_control_failure_diagnosis_report.v1.json`](../../outputs/20260910_v2/runs/lightgen_2d_semantic_real_path_control_failure_diagnosis_v1_attempt_01/lightgen_2d_semantic_real_path_control_failure_diagnosis_report.v1.json)，
SHA-256=`8b98dbcb2955417ca4dacd2298144bfdc006fd697dc1dc36eb49ff63dadcda86`；lineage 为
[`lightgen_2d_semantic_real_path_control_failure_diagnosis_lineage.v1.json`](../../outputs/20260910_v2/runs/lightgen_2d_semantic_real_path_control_failure_diagnosis_v1_attempt_01/lightgen_2d_semantic_real_path_control_failure_diagnosis_lineage.v1.json)，
SHA-256=`ce3f8a5101a300a764d4d005f9ca2eecfccc4db3c4a1a2dd1ca3860e511ba914`。

train/dev/calibration 原始图像 recall=`0.998/0.990/0.987`，real-path recall=`0.158/0.170/0.16775`；
calibration real/prior mean-power ratio=`2.2118e-4`，gauge-invariant effective rank=`16.4476411`，
类间中心/类内距离比=`0.9402647`，支持 encoder/complex-OLS semantic collapse。raw CMOS 非数值塌缩，
same-`W_OLS` 等价误差为 0，image-vs-latent gradient cosine=`0.0349641`、latent/image norm ratio=`0.0257622`。

diagnosis 未重新 forward protected/test，未训练、未更新参数、未生成 prior 样本图像、未调 evaluator/threshold、
未复用 sealed。下一动作是停止在该 bounded diagnosis；若继续，只能另行定义针对 encoder/OLS semantic collapse
的修正 training contract，不自动训练。

## 当前 definition 结果：LightGen 2D input/OLS alignment（2026-09-10）

契约为
`outputs/20260910_v2/runs/lightgen_2d_semantic_generation_input_ols_alignment_contract_v1_attempt_01/`
`lightgen_2d_semantic_generation_input_ols_alignment_contract.v1.json`，SHA-256=
`6f3859d87e9e3f38d141943ae3fac23607cc446fe2c58b99c9b5b4c379a3a72e`，status/result=`valid_completed/READY`，
classification=`DEFINE_LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_CONTRACT_V1_READY`。

契约冻结两条收敛路径：`X → explicit amplitude SLM mapping → illuminated complex field → input relay →
encoder → z_real∈C^100 → W_OLS → generator → raw CMOS`，以及
`z_generated∈C^100 → W_OLS → same generator → raw CMOS`。OLS 使用稳定 row-major `j=10*r+c`，
`z_j=<E_encoder,u_j>`，系数空间 `T=I_100`，强制 `W_OLS(z)=sum_j(Tz)_j v_j`；q 只作派生统计，
不能成为最终 latent。现有 `FiberArrayOLS`、`FiberModeBankV2`、`CoherentImagingRelay`、平面/复场类型
和 phase-only encoder/generator 作为可复用骨架，但缩尺度 LP01、sqrt 映射、端口顺序和 T 均标为项目决策，
不是 Chen 事实；整幅场 identity、label+8D、电子 decoder、旧 amplitude-first phase repair 保持历史或兼容基线，
不进入对齐主线。

本次 definition-only 未执行 CUDA、training、heldout scientific evaluation、generation acceptance 或 formal
promotion；未加载/复用 sealed 数值。`training_authorized=false`、`formal_promotion=false`、
`generation_readiness=not_ready` 保持。

## 当前 runtime 实现结果：LightGen 2D input/OLS alignment（2026-09-10）

实现结果为 `LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_RUNTIME_V1_IMPLEMENTED`。runtime 为
[`lightgen_2d_semantic_generation_input_ols_alignment.py`](../../src/ldm_project/v2/application/research/lightgen_2d_semantic_generation_input_ols_alignment.py)，
实现 SHA-256=`d01a234f6486a99c7ac2307b1b4021ed2ffccb53812a0a858675617a9023c3ce`；结构性验证报告为
[`lightgen_2d_semantic_generation_input_ols_alignment_runtime_report.v1.json`](../../outputs/20260910_v2/runs/lightgen_2d_semantic_generation_input_ols_alignment_runtime_v1_attempt_02/lightgen_2d_semantic_generation_input_ols_alignment_runtime_report.v1.json)，
SHA-256=`e26d24654e1d9c8707e9a6b93b5c5ee66f1172fb0ae7cd686d24ea03c9263af4`。

`forward_from_image(X)` 与 `forward_from_ols(z)` 已收敛到同一 canonical complex `[B,100]` OLS、同一
`w_ols`、同一 generator object/state 和 raw CMOS `|E_out|²`；projection、coefficient-space `T=I_100`、
modal re-emission 与 physical-plane checks 均为强制边界。7 个新测试及 Fiber/optics/LightGen 相关 30 个
回归测试通过；Ruff、py_compile、`git diff --check` 通过。验证只使用 CPU 非科学 smoke；该阶段未训练、未执行 heldout
generation、未读取 sealed。随后正式 training 已在独立明确授权下完成；`formal_promotion=false`、
`generation_readiness=not_ready` 保持。training contract、consumer、long-training preflight 与 formal training
已分别为 READY/IMPLEMENTED/AUTHORIZABLE/VALID_COMPLETED；当前唯一待决动作是定义 post-training evaluation
contract，不得自动开始评估或生成。

## 当前 definition 结果：LightGen 2D semantic complex-OLS training（2026-09-10）

definition-only consumer 已完成一次且仅一次，结果为 `valid_completed/READY`，classification 为
`DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONTRACT_V1_READY`。契约 artifact 为
[`lightgen_2d_semantic_complex_ols_training_contract.v1.json`](../../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_training_contract_v1_attempt_01/lightgen_2d_semantic_complex_ols_training_contract.v1.json)，
SHA-256=`7d384fe9a73acc8f883badc4973926ab0e58bfce737067d4764deef10ee1d16f`。

契约冻结：完整 `z∈C^100` 是唯一权威 latent；real encoded OLS 与 generated OLS 共用同一
`W_OLS` 与同一 generator；projection、coefficient-space `T=I_100` 和 modal re-emission 必须保留；
仅 `encoder.phase`、`generator.phase` 可训练，其余 OLS/relay/modal/evaluator 边界冻结。未来目标是
native complex isotropic Gaussian；latent 使用固定 full-complex gauge-invariant MMD，图像使用 raw CMOS
intensity MSE；不使用 q-only latent、q→phase reconstruction、历史 phase producer、电子 decoder 或 sealed 数值。

该 READY 仅表示后续训练 consumer 的定义完整，不是自动训练授权。随后用户已对唯一 scope
`EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V1` 作出明确授权，正式 training 已完成；definition
artifact 的 `training_authorized=false` 未被改写，`formal_promotion=false`、`generation_readiness=not_ready`
仍保持。heldout scientific evaluation、generation acceptance 与 promotion 尚未执行。

## 当前 consumer/preflight 结果：LightGen 2D semantic complex-OLS training（2026-09-10）

按 hash-bound training contract 完成一次且仅一次实现与 CPU engineering preflight，结果为
`valid_completed/IMPLEMENTED`，classification=`LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONSUMER_V1_IMPLEMENTED`。
consumer 为
[`lightgen_2d_semantic_complex_ols_training.py`](../../src/ldm_project/v2/application/research/lightgen_2d_semantic_complex_ols_training.py)，
入口为
[`execute_lightgen_2d_semantic_complex_ols_training_v1.py`](../../scripts/research/execute_lightgen_2d_semantic_complex_ols_training_v1.py)。

preflight artifact 为
[`lightgen_2d_semantic_complex_ols_training_preflight_input.v1.json`](../../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_training_consumer_v1_attempt_01/lightgen_2d_semantic_complex_ols_training_preflight_input.v1.json)，
SHA-256=`c16dc2b5a1ed1c8a4f0ba0fc2e5a3e90154620e3f053d54d79342cc434e85aa7`。3 个真实 train-role CPU 更新通过：
仅 `encoder.phase`、`generator.phase`（40,000 个 float32 参数）可训练，modal projection 与 `W_OLS` 均在实际
autograd path；full-complex prior、固定 gauge-invariant MMD、raw CMOS MSE、total loss decomposition、Adam
manifest、冻结状态与梯度防火墙均通过。

smoke checkpoint 为
`lightgen_2d_semantic_complex_ols_training_smoke_checkpoint.v1.pt`，SHA-256=
`6df6254196e24dec1cd7f902ca178de07efa022634c380326765a29250bce7be`；它仅用于 safe reload/resume 工程验证，
明确 `accepted_training_checkpoint=false`，不可用于 promotion。save→reload 后参数、optimizer、CPU/prior RNG、
sampler continuation 与 next-step 结果均一致。

该 consumer/preflight 阶段未执行 100-step future training、CUDA long compute、heldout evaluation 或 generation；
official MNIST test split 与 consumed N3 sealed evidence 均未读取/复用。随后正式训练已在独立明确授权下完成；
该 smoke checkpoint 仍不可晋升。

## 当前 long-training preflight 结果：LightGen 2D semantic complex-OLS（2026-09-10）

按 READY training contract 完成一次且仅一次
`PREFLIGHT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_V1`，结果为
`valid_completed/AUTHORIZABLE`，classification=`LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_AUTHORIZABLE`。
报告为
[`lightgen_2d_semantic_complex_ols_long_training_preflight.v1.json`](../../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_long_training_preflight_v1_attempt_01/lightgen_2d_semantic_complex_ols_long_training_preflight.v1.json)，
SHA-256=`365573edb0a3b4525bb571e31241bba6ba4243b9b2a3f591a8d1aea722c48d8a`；绑定 training contract SHA=
`7d384fe9a73acc8f883badc4973926ab0e58bfce737067d4764deef10ee1d16f`、consumer input SHA=
`c16dc2b5a1ed1c8a4f0ba0fc2e5a3e90154620e3f053d54d79342cc434e85aa7` 及当前实现身份。

目标为单张 `NVIDIA GeForce RTX 3060 Laptop GPU`（compute capability=`8.6`，VRAM=`6441926656` bytes），
production grid=`2048`、exact logical batch=`10`；NVRTC/complex exp/FFT/polar probes 通过，3 个真实 CUDA
forward/loss/backward/optimizer-step updates 通过，modal projection 与 `W_OLS` active，仅两个 phase trainable groups
更新，冻结状态未变。forward peak allocated=`2753601536` bytes，backward peak allocated=`2461667840` bytes，
minimum observed free=`2436896624` bytes。

target-device smoke checkpoint 为
[`lightgen_2d_semantic_complex_ols_long_training_preflight_smoke_checkpoint.v1.pt`](../../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_long_training_preflight_v1_attempt_01/lightgen_2d_semantic_complex_ols_long_training_preflight_smoke_checkpoint.v1.pt)，
SHA-256=`2ad10fd883cd4044627a7ae0d4ddc42754e3510e9dc13c78edfcb2ef6487d8aa`；它保持
`smoke_only=true`、`accepted_training_checkpoint=false`。save→reload→continuation 等价，next-step loss/parameter
diff=`0.0`；non-identity coefficient transport 被 `T=I_100` 防火墙拒绝。仅读取 train/dev，official test、heldout 与
sealed 均未读取/复用。

该结果只证明长训练在当前设备、当前 hash-bound contract 与当前资源边界上可授权，不是训练结果或生成结论；
随后正式训练已在独立明确授权下完成。当前 `formal_promotion=false`、`generation_readiness=not_ready`；训练合同
中的 `training_authorized=false` 定义标志未被改写。

## 当前 formal training 结果：LightGen 2D semantic complex-OLS（2026-09-10）

按用户对唯一 scope `EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V1` 的明确授权，使用冻结 training
contract、已通过的 target-device preflight 与当前 production consumer，完成一次且仅一次正式 CUDA training。
结果为 `VALID_COMPLETED`，status=`valid_completed`，classification=
`LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V1_VALID_COMPLETED`；100/100 updates 完成，运行约 `67.36 s`。

report 为
[`lightgen_2d_semantic_complex_ols_training_report.v1.json`](../../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_training_v1_attempt_01/lightgen_2d_semantic_complex_ols_training_report.v1.json)，
SHA-256=`f3f5fa876567513ec70cac67a6888aad6b55d140299992a535d0f9656bb4aa06`；lineage 为
[`lightgen_2d_semantic_complex_ols_training_lineage.v1.json`](../../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_training_v1_attempt_01/lightgen_2d_semantic_complex_ols_training_lineage.v1.json)，
SHA-256=`9c6eacb921b59788062236d9ba8354529f97bcf86de23813ba594e0fb7582c12`。目标为
`NVIDIA GeForce RTX 3060 Laptop GPU`、compute capability=`8.6`，production grid=`2048`、exact logical
batch=`10`。

只有 `encoder.phase`、`generator.phase` 可训练，共 `40,000` 个参数；full complex `z∈C^100`、modal
projection、`T=I_100`、`W_OLS`、full-complex gauge-invariant MMD 与 raw CMOS MSE 均按冻结合同生效。
所有 trace finite，冻结状态未变。初始 train `L_total=0.18129613995552063`，最终为
`0.15601420402526855`；step-50 dev `L_total=0.12443317954406738`，step-100 dev
`L_total=0.12301895964050293`，按固定规则选择 step 100。

step-50 checkpoint 为
[`lightgen_2d_semantic_complex_ols_training_checkpoint_step_050.v1.pt`](../../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_training_v1_attempt_01/lightgen_2d_semantic_complex_ols_training_checkpoint_step_050.v1.pt)，
SHA-256=`c467749eba3199fa4bc3815c76119a68571cac20167ea05cc735b1b04b1c23ff`；step-100 checkpoint 为
[`lightgen_2d_semantic_complex_ols_training_checkpoint_step_100.v1.pt`](../../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_training_v1_attempt_01/lightgen_2d_semantic_complex_ols_training_checkpoint_step_100.v1.pt)，
SHA-256=`6cfcc82831b575d83c097648a78b6507b32900e578d00c446d76ccd29130e149`。两者均已 safe-load 验证；
step-100 只是 selected training checkpoint，尚未接受为 formal/generation checkpoint。

数据仅使用 frozen train/dev roles；official test、heldout、sealed 均未读取或复用，post-training evaluation、
scientific generation、formal promotion 与 generation readiness promotion 均未执行。当前下一动作仅定义
`DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_POST_TRAINING_EVALUATION_CONTRACT_V1`，不自动评估或生成。

本次已按既有唯一授权动作 `EXECUTE_N3_COMPLEX_OLS_AMPLITUDE_PHASE_JOINT_COMPATIBILITY_V1`
完成一次且仅一次 CPU-only formal diagnostic execution。report 为
`outputs/20260910_v2/runs/n3_complex_ols_amplitude_phase_joint_compatibility_v1_attempt_01/`
`n3_complex_ols_amplitude_phase_joint_compatibility_report.v1.json`，SHA-256=
`9989b89dea8a8d299da9a89b8c5566376a27a21c250a5ca22c07202dfa938429`；lineage 为同目录
`n3_complex_ols_amplitude_phase_joint_compatibility_lineage.v1.json`，SHA-256=
`238d6638a16fcfa86e48216db2b886b45c9038f8ea017b7f51a55c1f18090aa0`。

不可选 control `FULL_COMPLEX_ROW_EMPIRICAL_BOOTSTRAP_CONTROL` 为 `5/5`；唯一 candidate
`OBSERVED_PHASE_INDEPENDENT_INHERITANCE_V1` 为 `0/5`，因此当前分类为
`N3_COMPLEX_OLS_INDEPENDENT_PHASE_INHERITANCE_NOT_SUPPORTED`。这只拒绝当前独立相位继承的
联合兼容性假设，不证明不存在 q-conditioned phase model，也不构成论文复现、物理失败或
generation 成功。selected producer 保持 `EMPIRICAL_GAUSSIAN_COPULA`；不复用已消费 sealed，
不把未披露的相位策略、prior 参数和 loss 实现猜测成事实。

definition-only 合同为
`outputs/20260910_v2/runs/n3_minimal_q_conditioned_phase_coupling_contract_v1_attempt_01/`
`n3_minimal_q_conditioned_phase_coupling_contract.v1.json`，SHA-256=
`14f4f3cd5d1046eb648b3410593d778910f8b8d33262528f9e0f12ba94250434`，
status/result=`valid_completed/READY`，classification=
`DEFINE_ONE_MINIMAL_Q_CONDITIONED_PHASE_COUPLING_MODEL_CONTRACT_V1_READY`。
合同只冻结项目模型假设 `Q_CONDITIONED_NEAREST_PHASE_DONOR_V1`、`K=1`、fit-derived
Gaussianized q Euclidean distance、稳定身份 tie-break 和完整相位向量复制；未来 evaluator
保持 `B=999`、per-fold alpha=`0.01`、strict `5/5`。当前机器 governance 未被该 definition-only
动作修改；该合同随后已按唯一授权动作消费一次，执行结果见下方 report/lineage。未自动训练或生成样本。

## 当前执行结果：q-conditioned nearest phase donor（2026-09-10）

执行动作 `EXECUTE_N3_Q_CONDITIONED_NEAREST_PHASE_DONOR_V1` 已完成一次且仅一次。冻结合同 SHA 为
`14f4f3cd5d1046eb648b3410593d778910f8b8d33262528f9e0f12ba94250434`；report 为
`outputs/20260910_v2/runs/n3_q_conditioned_nearest_phase_donor_v1_attempt_01/n3_q_conditioned_nearest_phase_donor_report.v1.json`
（SHA=`d17a472558349b043da07bc51310e7a350263608c059f37361237b1000318bcb`）；lineage 为
`outputs/20260910_v2/runs/n3_q_conditioned_nearest_phase_donor_v1_attempt_01/n3_q_conditioned_nearest_phase_donor_lineage.v1.json`
（SHA=`735fc652d334b2c4cc2a7b70a57c100d242149a17f72bb9d82a2de5b92f79c67`）。

结果为 `VALID_COMPLETED`，唯一分类为
`N3_COMPLEX_OLS_Q_CONDITIONED_NEAREST_PHASE_DONOR_NOT_SUPPORTED`。full-complex control
`FULL_COMPLEX_ROW_EMPIRICAL_BOOTSTRAP_CONTROL` 的 `5/5` 作为已验证 authority 原样复用、未重跑；
candidate 为 `0/5`。五折 calibration 均 valid，`exceedance=0`、`p_MC=0.001`，因此每折均 fail：

| fold | T_candidate | exceedance | p_MC | pass |
| ---: | ---: | ---: | ---: | :---: |
| 0 | 0.1324989195 | 0 | 0.001 | false |
| 1 | 0.1422853270 | 0 | 0.001 | false |
| 2 | 0.1595676023 | 0 | 0.001 | false |
| 3 | 0.1545405207 | 0 | 0.001 | false |
| 4 | 0.1536020389 | 0 | 0.001 | false |

本次严格保持 fitting-only donor pool、K=1、fit-derived Gaussianized-q Euclidean distance 与完整
100-D relative-phase vector inheritance；donor/phase/complex secondary diagnostics 仅为
`DESCRIPTIVE_ONLY`。sealed 不重用，selected producer 不变，`training_authorized=false`、
`formal_promotion=false`、`generation_readiness=not_ready` 保持；machine governance 未修改。

## 当前 definition 结果：complex-phasor linear residual bootstrap（2026-09-10）

definition-only 动作 `DEFINE_ONE_HIGHER_EXPRESSIVITY_Q_CONDITIONED_PHASE_MODEL_CONTRACT_V1` 已完成一次且仅一次。
合同为
`outputs/20260910_v2/runs/n3_q_conditioned_complex_phasor_linear_residual_bootstrap_contract_v1_attempt_01/`
`n3_q_conditioned_complex_phasor_linear_residual_bootstrap_contract.v1.json`，SHA-256=
`8f19c3dbb6b9bc17560276e7f1cf61004fd4b0583a3c8a15792b4a86fa05c29f`；status/result=`valid_completed/READY`，
classification=`DEFINE_ONE_HIGHER_EXPRESSIVITY_Q_CONDITIONED_PHASE_MODEL_CONTRACT_V1_READY`。

唯一 candidate 为 `Q_CONDITIONED_COMPLEX_PHASOR_LINEAR_RESIDUAL_BOOTSTRAP_V1`：同折 fitting-only
Gaussianized-q `x` 进入固定 `[1,x]`、101 列的 multivariate complex-phasor OLS；目标是完整 100-D
`U=exp(i*phi)`，不回归 wrapped angle。solver/rank 语义冻结为 CPU float64 thin SVD pseudoinverse、
`rcond=1e-12`、rank<101 即 `HARD_STOP`；残差是完整 100-D complex vector，uniform with replacement
抽取一个，unit projection 阈值 `abs(V)<=1e-12` 或 nonfinite 即 `INVALID_HARNESS`，无 retry/fallback。
`A(q)=q`、既有 gauge、当前 evaluator、`B=999`、per-fold alpha=`0.01` 和 strict `5/5` 均保持不变。

本次未加载 q/complex/sealed values，未执行 heldout evaluation、CUDA、training 或 generation；machine governance
未修改，`training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready` 保持。下一唯一动作是
`EXECUTE_N3_Q_CONDITIONED_COMPLEX_PHASOR_LINEAR_RESIDUAL_BOOTSTRAP_V1`，不自动开始。

## 当前执行结果：complex-phasor linear residual bootstrap（2026-09-10）

执行动作 `EXECUTE_N3_Q_CONDITIONED_COMPLEX_PHASOR_LINEAR_RESIDUAL_BOOTSTRAP_V1` 已完成一次且仅一次。
冻结 contract SHA=`8f19c3dbb6b9bc17560276e7f1cf61004fd4b0583a3c8a15792b4a86fa05c29f`；report 为
`outputs/20260910_v2/runs/n3_q_conditioned_complex_phasor_linear_residual_bootstrap_v1_attempt_01/`
`n3_q_conditioned_complex_phasor_linear_residual_bootstrap_report.v1.json`（SHA=`ae60752a27d3fb655b9354ace5591cb7f092db2aebb6206774853fa82cd025fe`）；
lineage 为同目录 `n3_q_conditioned_complex_phasor_linear_residual_bootstrap_lineage.v1.json`
（SHA=`503c5afa463403d237768e5eee1c5e130939394486e1c8348716ef553041a9de`）；candidate artifact
SHA=`e944a96e4f751db35ec84440b779cb7a8825145a78ea725703f839e3c4f9c587`。

结果为 `VALID_COMPLETED`，唯一分类为
`N3_COMPLEX_OLS_Q_CONDITIONED_COMPLEX_PHASOR_LINEAR_RESIDUAL_NOT_SUPPORTED`。固定 `energy_distance`、
`B=999`、per-fold alpha=`0.01`、strict `5/5` 下，五折均 rank=`101/101`、projection invalid=`0`，
但 exceedance=`0`、`p_MC=0.001`，candidate=`0/5`。full-complex control=`5/5` 作为已验证 authority
复用且未重跑；sealed confirmation `consumed=true` 仅作 provenance，`values_loaded=false`、`reused=false`。

该结果只拒绝固定 complex-phasor linear conditional mean + complete-vector residual 假设，不拒绝所有
conditional phase model、LightGen 输入对齐、论文机制或物理 generation。`training_authorized=false`、
`formal_promotion=false`、`generation_readiness=not_ready` 与 machine governance SHA 保持不变。
下一唯一动作是 `PLAN_BOUNDED_ARCHITECTURAL_DECISION_DIRECT_COMPLEX_JOINT_VS_LIGHTGEN_INPUT_ALIGNMENT`。

本次补写当前状态、交接和实验历史的正式 execution 结果，并新增 execution consumer/test 与 immutable
candidate artifacts；保留已有研究修改，不改 machine governance、runtime、数据、selected producer 或训练权限。
验证要求：报告/lineage/artifact hash、五折结构、control reuse、治理 flags、文档相对链接与 targeted tests
一致；无训练或 CUDA 验证。Sources 未 refresh/上传，云端旧摘要不能作为本次当前状态。

## 历史交接记录

以下完整保留整理前正文。内部“当前”“下一步”和各阶段 flags 都属于原记录时点，
不覆盖上方目标增量、当前状态页或机器证据。后续交接只更新本页上方，逐次实验正文进入实验历史。

<details>
<summary>展开整理前交接（历史状态与证据导航）</summary>

本文件是当前执行状态的稳定入口。正式 reconstruction mainline 已在当前 evidence boundary
收口，但独立的 OLS-native research line 已完成 N3 copula failure attribution 的 development-only
测量、定义并执行一次 raw-q mixed component-swap diagnostic，并冻结了 oracle-guided minimal
intervention contract；selector calibration 自身仍处于 MIXED 边界，N4 q-producer comparison
也已在冻结的 V2 authority 上完成且未支持任何候选；其后独立的 fixed-R Student-t copula
producer comparison 也已完成但未达到严格五折支持。只有下面的
“当前决策摘要”定义当前允许动作；其后的带日期或明确标为历史的条目是不可覆盖的历史 provenance，其中的
“下一步”只表示当时的记录，不得直接执行。架构合同见 [v2 runtime](../architecture/v2_runtime.md)，
假设边界见 [capacity-map hypothesis ledger](capacity_map_hypothesis_ledger.md)，完整历史事实见
[experiment history](experiment_history.md)。2026-08-24 重整前的长篇记录保存在
[archived handoff](../archive/current_handoff_pre_sced_20260824.md)，仅供溯源。

## 当前决策摘要

- 正式运行时基线：v2 `conditional_identity_logit_amplitude_v2`；正式 promotion、generation
  readiness 和 current training authorization 均未建立。
- 活动研究主线：`research/lightgen-ols-native-semantic-generation-v1`（精确 HEAD 与
  worktree 以本地 Git/bridge manifest 核验；本轮 comparison consumer、report 与状态文档变更后尚未
  refresh Sources，不声明 publication 为 `current_clean`）；N1/N2 完成，N3 producer
  baseline、selector calibration、两候选 minimal producer comparison 与 frozen local-location
  mixture selection 已完成但均未支持新 producer family；新的 local-shrinkage-covariance contract
  已执行一次 development-only selection，但结果为 inconclusive 且没有 selected K；N3 copula
  failure attribution 已完成但 aggregate 为 `N3_COPULA_FAILURE_MARGINAL_AND_DEPENDENCE_MIXED`；
  oracle-guided minimal intervention comparison V1 随后在 Candidate M 的冻结非负输出边界以
  `INVALID_HARNESS` blocked，未产生 scientific classification 或 selected candidate；V2 支持
  修复 contract 已冻结，并已由一次 fresh M2-vs-D comparison 消费完成；M2 为 `0/5`、D 为
  `1/5` compatible folds，aggregate 为 `NO_MINIMAL_INTERVENTION_SUPPORTED`，没有 selected
  candidate；随后 empirical-fit joint row-bootstrap ceiling diagnostic 通过 `3/5` folds，
  aggregate 为 `N3_FIT_HELDOUT_EMPIRICAL_CEILING_NOT_SUPPORTED`；再后续的 development-sampling
  stationarity diagnostic 判定 `N3_DEVELOPMENT_SAMPLING_STABILITY_LIMITATION_SUPPORTED`；随后
  冻结 class/source-position-balanced protocol 并完成唯一一次 empirical ceiling validation，
  结果为 `SOURCE_POSITION_BALANCED_DEVELOPMENT_PROTOCOL_NOT_SUPPORTED`（`0/5`）；随后按
  bounded authorization 完成 replacement `N3_DEVELOPMENT_BANK_V2` 的 source/q/five-fold
  authority 冻结，并由唯一一次 V2 empirical ceiling validation 得到
  `N3_DEVELOPMENT_BANK_V2_VALIDATED`（`5/5`）。V2 随后被唯一一次 N4 q-producer
  development comparison 消费：三个候选均未达到 all-five-fold support（copula=`4/5`）；
  随后 fixed-R Student-t copula producer comparison 为 `4/5` compatible folds，也未达到严格
  `5/5`；最新 self-null calibration 为 `VALID_COMPLETED` 但 aggregate=
  `N3_Q_COMPATIBILITY_SELF_NULL_MIXED_OR_UNRESOLVED`；随后 guardrail calibration 已按独立
  合同完成，classification=`N3_Q_COMPATIBILITY_GUARDRAIL_CALIBRATION_READY`，V3 只保留
  `energy_distance` 为 HARD primary，其余七项为 `DESCRIPTIVE_ONLY`。随后 V3 frozen-candidate
  replay 在 exact historical banks 上得到 Gaussian=`5/5` development-supported、Student-t=`4/5`
  primary-failing；历史 V2 classification 保持不变。producer-family search 已停止，且已完成
  producer-selection/freeze contract definition；V3 replay 与该 contract 均不等于 training 或
  generation authorization，N5 blocked，N6 保持独立冻结。随后已完成 selected-producer
  full-development fit、sealed-confirmation contract definition 与一次正式 sealed-confirmation
  execution；结果为 `N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_PASSED`，sealed 已按 write-once
  规则消费且不可 reset/retry。随后已完成 definition-only 的 complete-complex-OLS
  amplitude/phase joint-compatibility contract；当前唯一允许动作以紧随其后的 contract 条目为准，
  不自动执行。
- 最新 complete-complex-OLS amplitude/phase joint-compatibility contract 为
  [`n3_complex_ols_amplitude_phase_joint_compatibility_contract.v1.json`](../../outputs/20260910_v2/runs/n3_complex_ols_amplitude_phase_joint_compatibility_contract_v1_attempt_01/n3_complex_ols_amplitude_phase_joint_compatibility_contract.v1.json)，
  SHA=`52e07263343d765db13022fbcf56f74e7431a9dced16f8bfa31ca07ecc6bd726`，status/result=
  `valid_completed/READY`，classification=`N3_COMPLEX_OLS_AMPLITUDE_PHASE_JOINT_COMPATIBILITY_CONTRACT_V1_READY`。
  它绑定 `PROJECT_COMPLEX_OLS_V1`/`R3_COMPLEX`、`OBSERVED_RELATIVE_PHASE_INHERITANCE_V1`、V2
  source/q/five-fold authority 与 assignment SHA=`43ee0ce28e143fffdf2bcac98394625ccd766166ea717f23db66ed8770ef18a9`。
  q→A 固定为 `A_j(q)=q_j=abs(c_j/sqrt(s_j))`，d=`100`；相位为完整向量 `atan2` 主值区间，
  exact-zero 为 `INVALID_HARNESS`，near-zero 不阈值化。control 固定为
  `FULL_COMPLEX_ROW_EMPIRICAL_BOOTSTRAP_CONTROL`，唯一 candidate 为
  `OBSERVED_PHASE_INDEPENDENT_INHERITANCE_V1`；先 control、后 candidate，primary 仅为 gauge-canonical
  `[Re,Im]` embedding 的 `energy_distance`，`B=999`、per-fold alpha=`0.01`、严格 `5/5`。
  定义阶段未读取 q/complex/sealed 内容；旧 complex OLS artifact 只有历史 1000 rows，V2 的 5000
  rows 只绑定未来确定性物化，未执行 joint evaluation。sealed 已 consumed=true 且不可复用；
  `training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready`。唯一下一动作是
  `EXECUTE_N3_COMPLEX_OLS_AMPLITUDE_PHASE_JOINT_COMPATIBILITY_V1`。
- 最新 V3 guardrail calibration contract 为
  [`n3_q_compatibility_guardrail_gate_contract.v1.json`](../../outputs/20260909_v2/runs/n3_q_compatibility_guardrail_gate_contract_v1_attempt_01/n3_q_compatibility_guardrail_gate_contract.v1.json)，
  SHA=`a87979f9689159eee6d769f10eb6b7e30bcdc9803c4b72b108225a6ea22935b4`；report 为
  [`n3_q_compatibility_guardrail_calibration_report.v1.json`](../../outputs/20260909_v2/runs/n3_q_compatibility_guardrail_gate_v1_attempt_01/n3_q_compatibility_guardrail_calibration_report.v1.json)，
  SHA=`7135b07fd532ee5dd2dba7900ad6028c9a6d558924691e4d65eac56330367789`。共完成 `2000`
  个 model-self trials；直接配对五折的 V3 pass rate 为 Gaussian `200/200=1.0`、Student-t
  `200/200=1.0`，达到冻结 `0.80` 目标。V3 artifact 为
  [`n3_q_compatibility_evaluator_v3.v1.json`](../../outputs/20260909_v2/runs/n3_q_compatibility_guardrail_gate_v1_attempt_01/n3_q_compatibility_evaluator_v3.v1.json)，
  SHA=`4fbc831e383897802134ea2ce1aaef6446629130a7c799630254a86b8dcff3cc`；历史 N4、Gaussian、
  Student-t classification 未改写，sealed 未消费，`selected_producer=null`、
  `training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready` 保持。
- 最新 V3 frozen-candidate replay report 为
  [`n3_q_compatibility_evaluator_v3_frozen_candidate_replay_report.v1.json`](../../outputs/20260909_v2/runs/n3_q_compatibility_evaluator_v3_frozen_candidate_replay_v1_attempt_01/n3_q_compatibility_evaluator_v3_frozen_candidate_replay_report.v1.json)，
  SHA=`cf6e6d62e6d2a7b35822d2d3727d08e5cbd59d51778bf794ec30ce297ffd0195`，status/result=
  `valid_completed/VALID_COMPLETED`，classification=`N3_Q_COMPATIBILITY_EVALUATOR_V3_FROZEN_CANDIDATE_REPLAY_COMPLETE`。
  对应 lineage SHA=`d49800c12ce5f7e4931aca301410d5f0c47ca301350324fdfae3a73526d19ee4`，绑定 20 个
  历史 fit/sample artifacts。Gaussian 的 V3 primary pattern=`[true,true,true,true,true]`，
  classification=`N3_EMPIRICAL_GAUSSIAN_COPULA_V3_DEVELOPMENT_SUPPORTED`；Student-t pattern=
  `[true,true,false,true,true]`，classification=`N3_FIXED_R_STUDENT_T_COPULA_V3_NOT_SUPPORTED`。
  只有 `energy_distance` 作为 HARD veto，其他七项为 `DESCRIPTIVE_ONLY`；没有 refit、new sampling、
  heldout tuning 或 sealed access。`development_supported_candidate` 仅记录 Gaussian，未执行
  post-hoc selection；`selected_producer=null`、`training_authorized=false`、`formal_promotion=false`、
  `generation_readiness=not_ready` 保持。随后已完成 producer-selection/freeze contract definition；
  治理入口当前进入 full-development-fit blocker，不自动 final fit 或 sealed consumption。
- 最新 selected-producer freeze contract 为
   [`n3_selected_producer_freeze_contract.v1.json`](../../outputs/20260909_v2/runs/n3_selected_producer_freeze_contract_v1_attempt_01/n3_selected_producer_freeze_contract.v1.json)，
   SHA=`4d8d3f5dc81a7eff2d92a0acde0a0d0639788bb0dee75e6d88db745f642b60a3`，status/result=
   `valid_completed/VALID_COMPLETED`，classification=`DEFINE_N3_SELECTED_PRODUCER_FREEZE_CONTRACT_V1_READY`。
   contract 只依据 V3 Gaussian=`5/5` support 选择 `EMPIRICAL_GAUSSIAN_COPULA`，不以 descriptive
   guardrails 排序；历史 V2 Gaussian/Student-t 的 `4/5 NOT_SUPPORTED` 分类保持不变。后续唯一
   authorized action 是 `EXECUTE_N3_SELECTED_PRODUCER_FULL_DEVELOPMENT_FIT_V1`：全量 `5000`
   development q rows、`500/class`、无 heldout/fold fit、无 labels conditioning，并冻结经验边际、
   Gaussianization、`R_emp`/shrinkage、数值失败规则以及 standalone state/sampling interface。
   定义时 `frozen_producer_state=null`，尚未发生 all-5000 fit、state serialization 或 new sampling；
   该定义时点的治理入口为 `active_objective=N3_SELECTED_PRODUCER_FREEZE_CONTRACT_V1_READY`、
   `hard_blocker=N3_SELECTED_PRODUCER_FULL_DEVELOPMENT_FIT_REQUIRED`。其后的完成态见下一条。
- 最新 selected-producer full-development fit lineage 为
  [`n3_selected_producer_full_development_fit_lineage.v1.json`](../../outputs/20260910_v2/runs/n3_selected_producer_full_development_fit_v1_attempt_01/n3_selected_producer_full_development_fit_lineage.v1.json)，
  SHA=`9244df01c95e44005207cb1dc188fc7de2961665e13055beebf55f0f5cdd2267`，status/result=
  `valid_completed/VALID_COMPLETED`，classification=`N3_SELECTED_PRODUCER_FULL_DEVELOPMENT_FIT_COMPLETED`。
  它绑定唯一 state
  [`n3_selected_empirical_gaussian_copula_producer_state.v1.pt`](../../outputs/20260910_v2/runs/n3_selected_producer_full_development_fit_v1_attempt_01/n3_selected_empirical_gaussian_copula_producer_state.v1.pt)，
  semantic authority=`N3_SELECTED_EMPIRICAL_GAUSSIAN_COPULA_PRODUCER_V1`，state SHA=
  `752db16087e7bdf8343e4499202182370b376a1a2d5925a383f411fd4a15bb88`。拟合只消费 V2
  development q 的全量 `5000` 行、`500/class`，不使用 fold/heldout、不以 labels 条件化；
  state safe-load、数值结构和 standalone sampler smoke 均通过。没有 full-fit compatibility
  evaluation、sealed access、optical/semantic generation、CUDA 或 training；
  `selected_producer=EMPIRICAL_GAUSSIAN_COPULA` 仅是 development-only state authority。
  该 fit entry 记录的 state 仍是 development-only；`training_authorized=false`、
  `formal_promotion=false`、`generation_readiness=not_ready`，sealed 仍为
  `RESERVED_NOT_MATERIALIZED/consumed=false`。其后已完成下一 Gate 的 definition-only
  contract，当前治理入口与唯一下一动作以紧随其后的 sealed-confirmation 条目为准。
- 最新 sealed confirmation contract 为
  [`n3_selected_producer_sealed_confirmation_contract.v1.json`](../../outputs/20260910_v2/runs/n3_selected_producer_sealed_confirmation_contract_v1_attempt_01/n3_selected_producer_sealed_confirmation_contract.v1.json)，
  SHA=`e0e29a62bce4eaaa48346ce65d1d7d6dcc2892df4d8e9ace73681579232c1200`，status/result=
  `valid_completed/READY`，classification=`DEFINE_N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_CONTRACT_V1_READY`。
  它绑定冻结 producer state SHA=`752db16087e7bdf8343e4499202182370b376a1a2d5925a383f411fd4a15bb88`、
  lineage SHA=`9244df01c95e44005207cb1dc188fc7de2961665e13055beebf55f0f5cdd2267` 与 sealed
  reservation manifest SHA=`6475474f157d31591a3b0252f27320323d84ea93803c61ee6a365420a8ebb08e`。
  contract 只读取 reservation metadata，冻结 exact 500-row/50-class membership、既有
  V2 source→q semantics、V3 HARD `energy_distance`、4×200 model-bank aggregation、
  sample-size-matched model-self `B=999`/`alpha=0.05` Monte-Carlo rule，以及 descriptive-only
  七指标；development q95 不复用。exposure 前 arming 条件未执行，sealed 仍为
  `RESERVED_NOT_MATERIALIZED/consumed=false`（定义时状态）。consumed transition 被定义为
  first protected-content read 前的 write-once `consumed=true`；exposure 后 failure 只能为
  `SEALED_CONFIRMATION_INVALID_CONSUMED`，不得 reset/retry。其后执行结果见下一条。
- 最新 sealed confirmation execution lineage 为
  [`n3_selected_producer_sealed_confirmation_lineage.v1.json`](../../outputs/20260910_v2/runs/n3_selected_producer_sealed_confirmation_v1_attempt_01/n3_selected_producer_sealed_confirmation_lineage.v1.json)，
  SHA=`a8e8dc3ed3b4c2a5eb274aa0d089e6697aed8fdda83bf513c36bfdf548210bb7`，status/result=
  `valid_completed/VALID_COMPLETED`，classification=`N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_PASSED`。
  该 execution 完成 `B=999` model-self calibration、冻结 `4×200` observed MODEL banks，
  在 `SEALED_CONFIRMATION_ARMED` 后精确消费 `500` 个 sealed rows（`50/class`）。V3 HARD
  primary `energy_distance` 为 `T_sealed=0.04642494058877489`，exceedance=`473`，
  `p_MC=0.474`（`alpha=0.05`）；marker `consumed=true` 且不可逆。calibration、MODEL-bank、
  sealed q 与 q-materialization manifest 均在同一 run 目录冻结；未修改 producer/evaluator，
  未使用 CUDA、optimizer、backward 或 training。治理为
  `selected_producer=EMPIRICAL_GAUSSIAN_COPULA`、`n3_amplitude_producer_confirmation=PASSED`、
  `training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready`。
  下一规划边界为 `DEFINE_N3_COMPLEX_OLS_AMPLITUDE_PHASE_JOINT_COMPATIBILITY_CONTRACT_V1`，
  不自动定义或执行。
- 最新 N3 selector calibration V2 为 `N3_ATTRIBUTION_SELECTOR_CALIBRATION_MIXED_V2`。
  独立 synthetic confirmation 为 NULL=`0/200` false positives、partition=`100/100`、
  radial=`100/100`、nonlinear=`82/100`、local=`78/100`；local 未达到预注册 `0.80`
  correct-rate target，因此不能宣布 calibrated。
- 最新 source-position-balanced protocol artifact 为
  [`n3_source_position_balanced_development_protocol.v1.json`](../../outputs/20260909_v2/runs/n3_source_position_balanced_development_protocol_v1_attempt_01/n3_source_position_balanced_development_protocol.v1.json)，
  SHA=`5700ebb3b4b163d6d911763ce70ef77ca85f1c3ef387b7b0fdaed153e36e0904`，assignment SHA=
  `07d91587683792ba307f35dc06761ef3c55be49c52f05d0ddab0c3912d54e4d8`；随后唯一一次新的
  empirical ceiling report 为
  [`n3_source_position_balanced_empirical_ceiling_report.v1.json`](../../outputs/20260909_v2/runs/n3_source_position_balanced_empirical_ceiling_v1_attempt_01/n3_source_position_balanced_empirical_ceiling_report.v1.json)，
  SHA=`f9f86842aa063e67264a5be54a242eb37e9403756c9001b1f8b9f5e653dfd989`，结果为 `0/5`，
  classification=`SOURCE_POSITION_BALANCED_DEVELOPMENT_PROTOCOL_NOT_SUPPORTED`。该 authority
  保持历史不可变；V2 replacement evidence 见下一条。
- 最新 replacement development authority 为 `N3_DEVELOPMENT_BANK_V2`：source manifest
  [`n3_development_bank_v2_source_manifest.v1.json`](../../outputs/20260909_v2/runs/n3_development_bank_v2_source_manifest_v1_attempt_01/n3_development_bank_v2_source_manifest.v1.json)，
  SHA=`4bc251af91b3a6cd0235fe113117443d871a564bd05634fa9e61b5ab0b36f812`，bank semantic
  SHA=`bdb5a11c6800b4a5d0a994d5448bbf7de3e29c7bf8c0709f7d34be1cb31c91aa`，`5000` rows/
  `500/class`，eligible=`48500`，excluded union=`11500`，all registered source overlaps=`0`。
  CPU q manifest SHA=`422d8c65b78bc459197ed46504399cb9c87ba8f7184eb2405e0faa85353995d8`，q
  artifact SHA=`fe48c8fdb9bb2bf732076fabdf475288baa6f7d628ca10039a83bbda54fcfcbf`，q tensor
  SHA=`d660fd8dc449725edab081d00c8f553f151029cddef9210360a7a6fad4281209`；五折 manifest
  SHA=`4898a0e477d583b9a9e60c83238e6d8524d43967efe24277a952089e8adc447e`，assignment
  SHA=`43ee0ce28e143fffdf2bcac98394625ccd766166ea717f23db66ed8770ef18a9`，每折 `4000/1000`
  fit/heldout、每类 `400/100`。唯一 V2 ceiling report 为
  [`n3_development_bank_v2_empirical_ceiling_report.v1.json`](../../outputs/20260909_v2/runs/n3_development_bank_v2_empirical_ceiling_v1_attempt_01/n3_development_bank_v2_empirical_ceiling_report.v1.json)，
  SHA=`731cf826dc081e7cc5629790ba2fb74946f7cc41f1a7f0035bf3a3ca96a31826`，classification=
  `N3_DEVELOPMENT_BANK_V2_VALIDATED`，compatible=`5/5`。该 authority 已被后续 N4
  q-producer comparison 消费；本报告仍 `selected_producer=null`，不授权 training、promotion
  或 generation。
- 对应 machine evidence：
  [`n3_attribution_selector_v2_calibration_report.v1.json`](../../outputs/20260908_v2/runs/lightgen_n3_attribution_selector_v2_calibration_attempt_01/n3_attribution_selector_v2_calibration_report.v1.json)。
  frozen spec SHA=`9ee5b7ad64221b323f1b182c2f4d74edddfad89599bdf30b352ef33cc5b4c25b`，
  report SHA=`77c0a9f632b6870579a338d84dd6650530849fa6b74892acb1345567ffa8dae5`。
- 不能用未校准 selector 重新解释真实 N3 failure，也不能据此选择模型族。MIXED 边界关闭本
  selector refinement line，不创建 V3。此前 V1 attribution contract consumer 在 q tensor/
  metrics 之前因三处未定义执行分支返回 `BLOCKED`；V2 已补齐这些分支并完成一次有效的
  read-only attribution measurement，结果为 `N3_COPULA_FAILURE_MARGINAL_AND_DEPENDENCE_MIXED`。
- `DEFINE_N3_COPULA_FAILURE_ATTRIBUTION_CONTRACT_V2` 已完成 delta-only completeness revision，
  仅补齐上述三处分支并验证 V1 primary semantics unchanged；随后已被唯一 attribution
  consumer 消费完成。Machine artifact 为
  [`n3_copula_failure_attribution_contract.v2.json`](../../outputs/20260908_v2/runs/n3_copula_failure_attribution_contract_v2_attempt_01/n3_copula_failure_attribution_contract.v2.json)，
  base V1 SHA=`333b51abe2a6b1f3e96558273d4b6037267aa80678199d3bf6d0c300bb3508df`，V2 SHA=
  `5c85c4e949e5850de337d776f6782bdc791493b333ab9dd7982269e9f214eb95`，classification 为
  `DEFINE_N3_COPULA_FAILURE_ATTRIBUTION_CONTRACT_V2_READY`，claim 为
  `N3_COPULA_FAILURE_ATTRIBUTION_CONTRACT_V2_DEFINED`。V2 不改变 V1 primary semantics；
  执行结果由上方唯一 report 记录，仍不自动授权 P1 execution、producer、training 或 generation。
- 最新 N4 q-producer development comparison 为
[`n4_q_producer_comparison_report.v1.json`](../../outputs/20260909_v2/runs/n4_q_producer_development_comparison_v1_attempt_02/n4_q_producer_comparison_report.v1.json)，
SHA=`393d84617ec7b1e6c60b8c654b859dc3afe822f5dee207ca93816e232fea78be`，status/result=`VALID_COMPLETED`，
classification=`N4_Q_PRODUCER_COMPARISON_NO_CANDIDATE_SUPPORTED`。shrinkage elliptical Gaussian
与 radial scale mixture 均为 `0/5`，empirical marginal Gaussian copula 为 `4/5`，fold 4
失败于 `norm_energy_distance` 与 `squared_norm_energy_distance`。registry SHA=
`0329b23675a7d306c88f26c80a4757ce83d39c8c68b718bf7a1d966f53f4f27b`，evaluator exposure=`15/15`，
adaptive exposure=`false`；fit/heldout boundary、artifact lineage 与 governance closure 已审计。
- N4 之后仅完成了一个 definition-only 合同冻结：
[`n3_empirical_marginal_fixed_r_student_t_copula_contract.v1.json`](../../outputs/20260909_v2/runs/n3_empirical_marginal_fixed_r_student_t_copula_contract_v1_attempt_01/n3_empirical_marginal_fixed_r_student_t_copula_contract.v1.json)，
SHA=`0f18dccf35758e590255fac07436eb968c866ddc11a3e19df7b02e25770be1d7`，classification=
`DEFINE_N3_EMPIRICAL_MARGINAL_FIXED_R_STUDENT_T_COPULA_CONTRACT_V1_READY`，唯一 family 为
`EMPIRICAL_MARGINAL_FIXED_R_STUDENT_T_COPULA_V1`。它仅把 copula 径向 law 改为固定
  `R_shrunk` 下拟合一个 fit-only global `nu`；empirical marginals、R、V2 authority、评估和
  严格 `5/5` gate 均冻结，定义 consumer 未加载 q、未读 heldout、未拟合/采样/评估；后续
  producer execution 使用了新的独立 Gate，结果见下一条。
- 最新 fixed-R Student-t producer execution 的 machine report 为
  [`n3_empirical_marginal_fixed_r_student_t_copula_producer_report.v1.json`](../../outputs/20260909_v2/runs/n3_empirical_marginal_fixed_r_student_t_copula_producer_v1_attempt_01/n3_empirical_marginal_fixed_r_student_t_copula_producer_report.v1.json)，
  SHA=`4a938b883d1a0cf3386b7e2a5f4c900a236cfed031364c4a1273ea3ebe6d77dd`，status/result=
  `VALID_COMPLETED`，classification=`N3_FIXED_R_STUDENT_T_COPULA_NOT_SUPPORTED`。五个 fold
  均 operationally valid，compatible pattern=`[true,true,false,true,true]`（`4/5`）；fold 2
  失败于 `energy_distance`、`marginal_mean_l2`、`marginal_energy_distance_mean`。五折 fit-only
  `nu` 为 `118.35972638524892`、`118.04632498600743`、`117.03308605630197`、
  `118.47406043505019`、`119.33122553612606`。对应 lineage manifest SHA=
  `f2f943dab288726db6aaf3fb6c90acfe516ac7869fa9203a2c2a083731cf88da`；evaluator exposure=`5/5`，
  sealed 未消费。该结果是有效完成后的 scientific failure，不是 invalid harness；relative
  change 相比 N4 Gaussian 仅为 descriptive，未触发 selected producer。
- 最新 q-producer compatibility self-null calibration 的 immutable contract 为
  [`n3_q_producer_compatibility_self_null_calibration_contract.v1.json`](../../outputs/20260909_v2/runs/n3_q_producer_compatibility_self_null_calibration_contract_v1_attempt_01/n3_q_producer_compatibility_self_null_calibration_contract.v1.json)，
  SHA=`32ba12a8f09a850f0126bf528741928a247d957cf0a2105bd4329d102a97662c`。contract 只包含
  `EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_AMPLITUDE_V1` 与
  `EMPIRICAL_MARGINAL_FIXED_R_STUDENT_T_COPULA_V1`，固定每 producer/fold `20` trials，
  pseudo-DATA=`1000`、DATA-vs-MODEL=`4×200`、MODEL-vs-MODEL=`4×(200,200)`，完整使用现有
  V2 raw-q evaluator、q95 与八个 scalar guardrails；threshold、RNG namespace、无 early-stop/
  retry 规则均在 simulation 前冻结。
- 唯一 self-null report 为
  [`n3_q_producer_compatibility_self_null_calibration_report.v1.json`](../../outputs/20260909_v2/runs/n3_q_producer_compatibility_self_null_calibration_v1_attempt_01/n3_q_producer_compatibility_self_null_calibration_report.v1.json)，
  SHA=`0e69db97c52cb6e72383d2cd24227a0e9937b4aafb88413af6a9216ce93860fb`，status/result=`valid_completed/VALID_COMPLETED`，
  classification=`N3_Q_COMPATIBILITY_SELF_NULL_MIXED_OR_UNRESOLVED`。Gaussian fold pass rates
  为 `[0.80,0.90,1.00,0.90,0.90]`、five-fold diagnostic product=`0.5832`；Student-t 为
  `[0.95,0.95,0.90,1.00,0.95]`、product=`0.7716375`；两者 primary ED false-fail rate 均为
  `0`，但 norm/squared-norm 与少量 marginal guardrails 发生 self-null false failures。
  这既不能宣布 evaluator stable，也没有达到预注册 false-fail limitation 类别；因而不把
  `4/5` failures 归因给任何一方，下一动作只允许针对该一个 mixed self-null uncertainty 做
  一个 bounded decision，不重复两个既有 producer、不评估第三 family。
- future producer-family confirmation 必须使用 fresh confirmation source；当前 internal-validation
bank 仅为 N3 development bank。当前 N4 classification 为
`N4_Q_PRODUCER_COMPARISON_NO_CANDIDATE_SUPPORTED`（此前 copula-only result
`N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_NOT_SUPPORTED` 保留为历史 evidence），
  `training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready` 与
  sealed confirmation `RESERVED_NOT_MATERIALIZED` 保持不变。oracle-guided M/D contract 已冻结，
  V1 comparison 曾在 Candidate M 的冻结非负输出边界处以 `INVALID_HARNESS` 阻塞；V2 comparison
  已按独立授权有效完成但未达到 5/5 fold support，因此这些 comparison 不选择 producer、OLS、
  generation 或 training；fixed-R Student-t execution 同样未打开任何下游边界，当前仍保持
  no-candidate blocker，不改变这些边界。

- `EXECUTE_N3_COPULA_FAILURE_ATTRIBUTION_V1` 的唯一 machine report 为
  [`n3_copula_failure_attribution_report.v1.json`](../../outputs/20260909_v2/runs/n3_copula_failure_attribution_v1_attempt_01/n3_copula_failure_attribution_report.v1.json)，
  SHA=`e9a834150baa006c619256b8306340e31593e1c9562a37ea46955b34ae86cac8`。冻结 `M` 为
  `0/2/3=1, 1/4=0`，五个 fold 的 `D` 均为 `1`；labels 为
  `0/2/3=MARGINAL_AND_DEPENDENCE`、`1/4=DEPENDENCE_ONLY`，故 V2 aggregate 为
  `N3_COPULA_FAILURE_MARGINAL_AND_DEPENDENCE_MIXED`。两个 pass-control folds 的 dependence
  failure 使其不能成为 dominant class；Spearman、spectral、support/tail 仅为 descriptive。
  该报告明确记录固定 1000-row development pool 的 reuse limitation，P1 execution 未授权。
- `DEFINE_N3_COPULA_MIXED_COMPONENT_SWAP_DIAGNOSTIC_CONTRACT_V1` 已完成 definition-only
  contract-totality 校验。Machine artifact 为
  [`n3_copula_mixed_component_swap_diagnostic_contract.v1.json`](../../outputs/20260909_v2/runs/n3_copula_mixed_component_swap_diagnostic_contract_v1_attempt_01/n3_copula_mixed_component_swap_diagnostic_contract.v1.json)，
  SHA=`d46c4d8e8d8dbc41f325f2e1d7f001d645479a60deef08e5d514a80bada22763`，classification 为
  `DEFINE_N3_COPULA_MIXED_COMPONENT_SWAP_DIAGNOSTIC_CONTRACT_V1_READY`。它只冻结历史
  DATA/MODEL pairing、ordinal tie/rank、`H_MARG`/`H_DEP`、原始 raw-q ED/q95、pair/fold/
  aggregate oracle-sufficiency mapping 与 claim boundary；未加载 q、未执行 swap、未计算
  oracle metric，不是 producer-support result，也不自动授权 diagnostic execution；定义本身与
  后续执行结果分离。
- `EXECUTE_N3_COPULA_MIXED_COMPONENT_SWAP_DIAGNOSTIC_V1` 已在上述 contract 下完成唯一一次
  CPU-only、development-only、read-only execution。Machine report 为
  [`n3_copula_mixed_component_swap_diagnostic_report.v1.json`](../../outputs/20260909_v2/runs/n3_copula_mixed_component_swap_diagnostic_v1_attempt_01/n3_copula_mixed_component_swap_diagnostic_report.v1.json)，
  SHA=`fc139cb6ec9754babeb9e4ac6c03f08ef339cbdd6148362bb05b44c08daeaecc`，Gate result 为
  `VALID_COMPLETED`，精确 aggregate=`N3_COPULA_PRIMARY_BOTH_ORACLES_SUFFICIENT`。Fold labels
  为 `0/1/2/3=BOTH_ORACLES`、`4=CONTROL_FOLD`；20/20 pair labels、9 个原始 primary fail
  pair 和 11 个 passing controls 均已输出，原始 fail pair 全部为 `BOTH_ORACLES`。5 个 DATA
  rank 与 60 个 MODEL rank 的 self-reconstruction/permutation/tie-order identity、20/20
  `E_BASE` 原始字段重建均通过；q95 只复用原始 fold-local report 值。所有 hybrids 都标注
  `DIAGNOSTIC_ONLY`、`ORACLE_USES_HELDOUT_DATA`、`NOT_A_PRODUCER`、
  `NOT_GENERATION_EVIDENCE`；无 RNG、无 sealed confirmation、无训练或生成动作，
  `p1_execution_authorized=false`。
- `DEFINE_N3_ORACLE_GUIDED_MINIMAL_INTERVENTION_CONTRACT_V1` 已完成 definition-only
  contract-totality 校验。唯一 machine artifact 为
  [`n3_oracle_guided_minimal_intervention_contract.v1.json`](../../outputs/20260909_v2/runs/n3_oracle_guided_minimal_intervention_contract_v1_attempt_01/n3_oracle_guided_minimal_intervention_contract.v1.json)，
  SHA=`d640335c298afed696c64250acd15ed4767e8222a8e962f54fe6cd05c8a0f6b1`，classification 为
  `DEFINE_N3_ORACLE_GUIDED_MINIMAL_INTERVENTION_CONTRACT_V1_READY`，claim 为
  `N3_ORACLE_GUIDED_MINIMAL_INTERVENTION_CONTRACT_DEFINED`。M 是只改 marginal 的
  `log1p` Gaussian location-scale family；D 是只改 dependence 的 fixed-`nu=5`
  Student-t copula，分别冻结 complementary component。复杂度 tuple 预先固定为 M=`5250`
  vs D=`85050` effective fitted scalar count，双方通过时选择 M，且没有 post-result score。
  后续仅允许一次固定五折 `800/200`、raw-q ED + 当前 joint/marginal guardrails 的 development
  comparison；本定义未 fit/sample/观察任何新候选结果，不授权 comparison、confirmation、
  N4/N5/N6、training 或 generation。
- `EXECUTE_N3_ORACLE_GUIDED_MINIMAL_INTERVENTION_COMPARISON_V1` 已完成静态 authority/lineage
  校验并进入一次 CPU-only execution，但在 Candidate M 的 fold `0`、`DATA-vs-MODEL`、replicate
  `0` 停止：`expm1` 输出中 `351/20000` 个值为负，`q_min=-0.42263710721584474`，而冻结
  contract 要求 finite/nonnegative 且禁止任何修复或拒绝采样。该情形按 contract 为
  `INVALID_HARNESS`，Gate result=`BLOCKED`；没有 scientific classification、compatible-fold
  count、selected candidate 或 partial candidate result，Candidate D 未执行。唯一 machine report
  为 [`n3_oracle_guided_minimal_intervention_comparison_report.v1.json`](../../outputs/20260909_v2/runs/n3_oracle_guided_minimal_intervention_comparison_v1_attempt_01/n3_oracle_guided_minimal_intervention_comparison_report.v1.json)，
  SHA=`ff8404e4c6b26b66d9292e728e30031cc5b14b7c1d1396d5713e6a2bdae77c17`。该 block 不授权修订
  contract、定义 V2、final all-development fit、sealed confirmation、P1、N4/N5/N6、training
  或 generation；sealed `RESERVED_NOT_MATERIALIZED`/`consumed=false` 与全局 flags 保持关闭。
- `DEFINE_N3_ORACLE_GUIDED_MINIMAL_INTERVENTION_CONTRACT_V2` 已完成一次 definition-only
  contract consumer，V1 contract 与 INVALID_HARNESS report 均按 exact SHA 绑定且保持 immutable。
  新 machine authority 为
  [`n3_oracle_guided_minimal_intervention_contract.v2.json`](../../outputs/20260909_v2/runs/n3_oracle_guided_minimal_intervention_contract_v2_attempt_01/n3_oracle_guided_minimal_intervention_contract.v2.json)，
  SHA=`f297a1ffbf84bec84a15525be97ba1a5fd897d32beefd0fe22138bff0012ce2e`，classification 为
  `DEFINE_N3_ORACLE_GUIDED_MINIMAL_INTERVENTION_CONTRACT_V2_READY`，claim 为
  `N3_ORACLE_GUIDED_MINIMAL_INTERVENTION_CONTRACT_V2_DEFINED`。M2 只将 M 的 marginal 改为
  fixed-boundary zero-truncated `log1p` Gaussian，保留 V1 `mu/sigma`、Gaussian copula、RNG
  与 comparison semantics；`q>=0` 的解析 support proof、`1e-12` floating-boundary policy、
  5 个 synthetic support cases 与 semantic diff 均通过。D 保持 byte/semantic-equivalent，
  complexity 为 M2=`5250`、D=`85050`，双方通过时 tie-break 为 M2。没有 fit/sample M2 或 D；
  定义 consumer 本身未 fit/sample；随后 fresh M2-vs-D comparison 已按独立授权完成，详见下一条
  execution record。sealed、training、promotion 和 generation flags 保持关闭。
- `EXECUTE_N3_ORACLE_GUIDED_MINIMAL_INTERVENTION_COMPARISON_V2` 已按上述 V2 contract 完成唯一
  一次 CPU-only、development-only、fresh fit/sample/evaluate execution。Machine report 为
  [`n3_oracle_guided_minimal_intervention_comparison_report.v2.json`](../../outputs/20260909_v2/runs/n3_oracle_guided_minimal_intervention_comparison_v2_attempt_01/n3_oracle_guided_minimal_intervention_comparison_report.v2.json)，
  SHA=`19d4d933f28200010373ed8210f31f343ae40d6474f6bdfb05e7fcb0c5fcf192`，Gate result=`VALID_COMPLETED`，
  classification=`NO_MINIMAL_INTERVENTION_SUPPORTED`，selected candidate 为 `null`。M2 与 D 的
  五折均 operationally valid；M2 compatible=`0/5`（material negative=`0`、roundoff
  canonicalization=`0`），D compatible=`1/5`，固定 `nu=5` 在五折均验证通过；D 仅 fold 3
  同时通过全部 8 个 guardrails 与 primary。比较未使用 relative/weighted selection，V1
  `INVALID_HARNESS` report 保持 immutable；未执行 final all-development fit、sealed confirmation、
  P1、N4/N5/N6、complex OLS、training 或 generation，global flags 保持关闭。
- `DEFINE_N3_EMPIRICAL_FIT_JOINT_BOOTSTRAP_CEILING_DIAGNOSTIC_V1` 已先完成 definition-only
  contract-totality；合同为
  [`n3_empirical_fit_joint_bootstrap_ceiling_diagnostic_contract.v1.json`](../../outputs/20260909_v2/runs/n3_empirical_fit_joint_bootstrap_ceiling_diagnostic_v1_attempt_01/n3_empirical_fit_joint_bootstrap_ceiling_diagnostic_contract.v1.json)，
  SHA=`fc24ca4c863fded5f23dacc69b13bb16f44224b96d47888d62cb9a3f0f70a38d`，定义阶段未加载 q。
  `N3_EMPIRICAL_FIT_JOINT_BOOTSTRAP_CEILING_DIAGNOSTIC_V1` 随后按同一 frozen five-fold
  `800/200` identity 完成唯一一次 CPU-only、development-only complete-row bootstrap；每折
  `4` 个 DATA-vs-MODEL 与 `8` 个 MODEL-vs-MODEL banks，8 个既有 scalar guardrails 全部适用，
  duplicate-row rate 仅作 descriptive。
- 唯一 machine report 为
  [`n3_empirical_fit_joint_bootstrap_ceiling_diagnostic_report.v1.json`](../../outputs/20260909_v2/runs/n3_empirical_fit_joint_bootstrap_ceiling_diagnostic_execution_v1_attempt_01/n3_empirical_fit_joint_bootstrap_ceiling_diagnostic_report.v1.json)，
  SHA=`c4557d44c8d1b40adeb2f362a4f81f0f28d9feb398428fbb1b13c035b043e12c`，Gate result=`VALID_COMPLETED`，
  classification=`N3_FIT_HELDOUT_EMPIRICAL_CEILING_NOT_SUPPORTED`，compatible=`3/5`。fold 0/2
  分别只因 `marginal_std_l2`/`marginal_energy_distance_max` 失败，fold 1/3/4 完整通过；
  五折均 operationally valid。FACT 是拟合经验联合分布本身未达到冻结 `5/5` held-out gate；
  MODEL_ASSUMPTION 是结果与 fit-to-heldout distribution stability / finite-development sampling
  limitation 一致，但不是无 population-level producer 的证明。该 diagnostic 不是 producer、
  confirmation 或 generation evidence；未执行 M3、新 copula、KDE、mixture、vine、flow/neural
  density、producer selection、final all-1000 fit、sealed、complex OLS、N4/N5/N6、CUDA 或
  training，global flags 保持关闭。下一规划目标为 development sampling/stationarity contract。
- `DEFINE_N3_DEVELOPMENT_SAMPLING_STATIONARITY_DIAGNOSTIC_V1` 已完成 definition-only
  contract-totality；合同为
  [`n3_development_sampling_stationarity_diagnostic_contract.v1.json`](../../outputs/20260909_v2/runs/n3_development_sampling_stationarity_diagnostic_v1_attempt_01/n3_development_sampling_stationarity_diagnostic_contract.v1.json)，
  SHA=`05c0c5cb318566d59e4d46da9693cede0cbf9958b69656510b6deaf3e6e18380`，未加载 q。合同绑定
  frozen 1000-row development authority、既有 source-position authority 与 empirical-ceiling
  fold linkage，冻结 raw-q mean/std/radial moment、1000 个 balanced random partitions、global q95
  与 class q99；官方五折保持不变，无新 producer。
- 该 diagnostic 的首次 execution attempt 在科学分类前因浮点负方差以 `INVALID_HARNESS`
  blocked，immutable report 保留；修复为直接 subset moment computation 后，合同语义和阈值未变。
  唯一有效 report 为
  [`n3_development_sampling_stationarity_diagnostic_report.v1.json`](../../outputs/20260909_v2/runs/n3_development_sampling_stationarity_diagnostic_execution_v1_attempt_02/n3_development_sampling_stationarity_diagnostic_report.v1.json)，
  SHA=`625b4edda39066a4c414337fc0ba7d06d64fda77b6d41d8241806c715ee871a3`，Gate result=`VALID_COMPLETED`，
  classification=`N3_DEVELOPMENT_SAMPLING_STABILITY_LIMITATION_SUPPORTED`。五折 global moment
  均在 finite-partition reference 内，fold 0/2 均非 exceptional；class q99 仅 fold 1 label 6
  为 descriptive exception，composition difference=`false`。限制性判定来自 registered source-position
  drift（pooled r2 Spearman=`0.4472886`、10/10 class signs positive），empirical-ceiling
  `false/true/false/true/true` linkage 未被重定义；下一动作是停止 producer-family escalation，先
  repair development sampling/stationarity protocol。
- `DEFINE_N3_SEALED_CONFIRMATION_SOURCE_V1` 已完成；reservation 状态为
  `RESERVED_NOT_MATERIALIZED`。Manifest 与完整 lineage 见 [machine manifest](../../outputs/20260908_v2/runs/n3_sealed_confirmation_source_v1_attempt_01/n3_sealed_confirmation_source_manifest.v1.json)。
- `COMPARE_N3_MINIMAL_COMPETING_PRODUCER_BASELINES_V1` 已在既有 1000-row development
  pool 上完成固定 5-fold、A/B 两候选 comparison；fold assignment SHA 为
  `90277c16a0d96b895e218264f35a822564b3a46570cab73a1a1e823f063e9e32`。两候选均可运行并
  达到固定 positivity bank，但均在 5/5 folds 超出自身 model-vs-model q95 primary reference
  与 covariance/norm guardrails，结果为
  `N3_MINIMAL_PRODUCER_BASELINES_NOT_SUPPORTED`，没有 selected candidate。Report SHA 为
  `af7611cfbca211e666d7773e81b230a1b380b91583817c3c4ab80925f2a16440`。
- `DEFINE_N3_UNSUPERVISED_LOCAL_LOCATION_MIXTURE_PRODUCER_CONTRACT_V1` 已完成并冻结
  shared-covariance local-mixture contract，候选 `K={2,4,8}`，复用同一 frozen fold assignment；
  contract classification 为
  `DEFINE_N3_UNSUPERVISED_LOCAL_LOCATION_MIXTURE_PRODUCER_CONTRACT_V1_READY`，SHA 为
  `38cb0c549d8f943123f0ab3304adf5a01e9b3b5edb63daa384c7f4c7bac20449`。这只是
  `N3_LOCAL_LOCATION_MIXTURE_MODEL_ASSUMPTION_DEFINED`，尚未 fit、sampling 或选择 K。
- `EXECUTE_N3_LOCAL_LOCATION_MIXTURE_SELECTION_V1` 已严格执行 frozen contract；K=2、4、8 均
  5/5 folds operationally valid，但均为 0/5 compatible，最终 classification 为
  `N3_LOCAL_LOCATION_MIXTURE_NOT_SUPPORTED`，没有 selected K。Report SHA 为
  `5064ee87cbc9846f0194ba0afcea4687647ceefb454d2abb20b8b56d632a9080`；这是 development-only
  valid failure，不授权 final all-1000 fit、sealed confirmation 或 generation。
- `DEFINE_N3_UNSUPERVISED_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_CONTRACT_V1` 已完成并冻结下一
  minimal family：保留 Stage-1 unsupervised local partition，Stage-2 对 assigned fitting rows
  使用 component-local analytic shrinkage covariance。classification 为
  `DEFINE_N3_UNSUPERVISED_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_CONTRACT_V1_READY`，SHA 为
  `2cb596abe61deeaa847b90c74654c5abe0e94aae18a6f252170f8149ad1c1550`。这只是
  `N3_LOCAL_COVARIANCE_HETEROGENEITY_MODEL_ASSUMPTION_DEFINED`，尚未 fit、sampling 或选择 K。
- `EXECUTE_N3_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_SELECTION_V1` 已严格执行一次。Stage-1 的
  15/15 candidate-fold 均收敛；Stage-2 的 `n_k<2` 在 K=2/4/8 分别造成 1/2/4 个 fold
  failure，剩余 operational folds 均未通过 primary/covariance/norm compatibility。最终为
  `N3_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_INCONCLUSIVE`、`K_selected=null`。Report SHA 为
  `cbd16bff7f4e382f6db702099e87da4468d3213d2959492001997e4e0fb1da5a`。
- 恢复 outputs、历史 evidence、producer、rank、partition、formal v2 和全局 flags 均保持不变；
  sealed confirmation 仍为 `RESERVED_NOT_MATERIALIZED`，`training_authorized=false`、
  `formal_promotion=false`、`generation_readiness=not_ready`。
- 当前没有自动后续 N3 family action：不得将 inconclusive 结果改写为 selected 或
  NOT_SUPPORTED，不得修改 Stage-1/Stage-2 合同、增加 K、改变 `n_k<2` 语义、消费 sealed
  source、启动 selector V3、generation 或重跑此前 local-location line。任何后续研究必须先有
  独立 planning Gate。
- `DEFINE_N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_PRODUCER_CONTRACT_V1` 已完成并冻结一个
  单一 candidate family：fit-only empirical marginal quantile functions 加一个 regularized
  Gaussianized-score dependence matrix。Contract SHA 为
  `3edc9fee006f68eff6ba41205b7e4e6b6cbea3205080c4cb333689e554fba7a4`，classification 为
  `DEFINE_N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_PRODUCER_CONTRACT_V1_READY`，claim 仅为
  `N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_MODEL_ASSUMPTION_DEFINED`；未执行 real fit、sampling、
  model selection 或 confirmation。未来 execution 需要独立明确授权，不得在看到结果后改变
  marginal、copula、shrinkage 或 compatibility semantics。
- `EXECUTE_N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_PRODUCER_V1` 已按该 contract 严格执行一次，
  report SHA 为 `78b8e40f21015af8d704a89a83bb6b4043c0208386032b41b1ced4dd9bdeba7d`，结果为
  `N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_NOT_SUPPORTED`。五个 fold 均完成 finite/PD/单位对角
  与 finite/nonnegative sampling 检查，但仅 fold 1、4 通过完整 compatibility；fold 0、2、3
  失败，故不满足 all-five-fold rule。描述性 marginal-vs-joint 结果为 `MIXED`：fold 0/3
  同时有 marginal/joint 残差，fold 2 主要剩余 marginal guardrail，fold 1/4 无残余失败。
  这不是 harness failure，也不改变此前 local-covariance `INCONCLUSIVE` 结果。
- 本次 execution 仍是 development-only；sealed confirmation 保持
  `RESERVED_NOT_MATERIALIZED`、`consumed=false`，不授权 final all-1000 fit、confirmation、
  novel OLS、generation 或 training。当前没有自动后续 producer family action；下一动作只能
  是独立 planning Gate，明确剩余 marginal/tail 与 joint-dependence mismatch 的研究问题。
- `RECOVER_FROZEN_N3_COPULA_BANK_AUTHORITY_V1` 已解除此前 attribution 的 numerical-bank blocker。
  既有 output roots 中没有可消费的完整 copula bank artifact；旧 r=99 tensors 未消费。一次且
  仅一次 exact deterministic rematerialization 成功，recovery manifest SHA 为
  `4ed6ac4839cdb075f98d07be3162ac560ea29cc260fcbd0e272a342aed04297c`，tensor artifact SHA 为
  `6b9da40cf868fb9033e5c45aa05778c27199867430d2ec5627b36875d284b42b`，60/60 q hashes 完全相等，
  semantic authority=`FROZEN_N3_COPULA_GENERATED_BANKS_V1`。该动作只恢复历史 sample identity，
  没有重算 metrics、修改原 report 或完成 attribution；sealed confirmation 仍为
  `RESERVED_NOT_MATERIALIZED`。随后已完成独立 contract definition；唯一后续科学消费者仍是
  另行授权的 `RESOLVE_N3_COPULA_FAILURE_MARGINAL_VS_DEPENDENCE_V1` read-only consumer。
- 前一 `RESOLVE_N3_COPULA_FAILURE_MARGINAL_VS_DEPENDENCE_V1` 因缺少 frozen
  attribution contract/spec/consumer 正确返回 `BLOCKED`；随后
  `DEFINE_N3_COPULA_FAILURE_ATTRIBUTION_CONTRACT_V1` 已完成并写入唯一 write-once machine
  artifact：[`n3_copula_failure_attribution_contract.v1.json`](../../outputs/20260908_v2/runs/n3_copula_failure_attribution_contract_v1_attempt_01/n3_copula_failure_attribution_contract.v1.json)，
  SHA=`333b51abe2a6b1f3e96558273d4b6037267aa80678199d3bf6d0c300bb3508df`。其 execution
  preflight 后来发现三处定义缺口并在 q tensor/metrics 前阻断；V1 未修改。现由上述 V2
  artifact 仅作完整性 supersession，V2 definition consumer 未加载 q tensor、未计算 metrics、
  未 fit/sampling/选择 producer，也未消费 sealed confirmation。

## 本轮 evidence I/O 修复（2026-09-08）

已修复 `src/ldm_project/v2/evidence/io.py` 中会把设备、资源或程序异常宽泛包装成
`ValueError` 的路径：只将 weights-only 的 `pickle.UnpicklingError` 转为明确的安全载荷拒绝，
其余异常保留原始类型和原因；canonical serialization 不再把任意对象的 `__dict__` 默认为
稳定 schema；artifact digest、shape、dtype 和非 tensor 输入仍 fail-closed。公共函数签名、
有效 artifact 的规范化结果和历史哈希兼容性保持不变。定向回归 `23 passed`、Ruff 定向检查
已通过；默认全套 pytest 已尝试（`1609 passed, 12 deselected, 25 subtests passed, 5 failures`），
失败项为既有分支/布局或临时路径环境问题；最终 `git diff --check` 已通过。

## 历史合同草案（已由 V2 执行结果取代）

`PLAN_N3_ATTRIBUTION_SELECTOR_CALIBRATION_REPAIR` 是已完成前的 planning contract，不是当前待批准动作。
其建议的 replicate、FPR、置信界限和机制控制已由后续
`CALIBRATE_AND_CONFIRM_N3_ATTRIBUTION_SELECTOR_V2` 明确采用并执行；权威事实、结果与限制集中见
[实验历史](experiment_history.md)。本节保留名称与取代关系，不能追溯修改 evidence，也不授予后续科学权限。

## Research governance simplification（2026-09-07）

`RESEARCH_GOVERNANCE_SIMPLIFICATION_V1_ESTABLISHED`：本轮只改变 research planning、evidence
handoff 与 provenance presentation；不改变 scientific result、formal/high-risk evidence、
training authorization 或 historical artifact。

普通交接采用 [`delta_one_gate_short_prompt.md`](../operations/delta_one_gate_short_prompt.md)
的 R0–R3 风险层、one-outer-Gate 和 semantic authority ID；完整 child Hash 只留在其
machine consumer。N3 的 `RESOLVE_N3_SCORE_GEOMETRY_ATTRIBUTION_V1` 与后续
`CALIBRATE_N3_ATTRIBUTION_DECISION_RULE_V1` 已完成；当时记录的下一步
`PLAN_N3_ATTRIBUTION_SELECTOR_CALIBRATION_REPAIR` 随后已由 V2 consumer 执行并取代，
不再是当前动作；当时仍不执行 fit、SVD、rank selection、sampling、forward、CUDA 或 generation。

## N3 研究历史导航

OLS-native N3 的 branch genesis、producer data authority、rank/fit/validation、mismatch attribution、
score-geometry resolution 与两轮 selector calibration 均已迁移到唯一的
[实验历史](experiment_history.md) 条目。本交接只保留当前摘要、阻塞与允许动作；历史条目中的下一步
不构成当前授权，恢复 outputs 和 machine evidence 仍按现有 immutable 边界读取。

## S3 历史研究导航

2026-09-07 的 controller semantic-geometry data authority、one-shot canary、frozen S3 consumer
与 postmortem 已集中记录在 [实验历史](experiment_history.md)。本交接只保留当前 N3 增量、阻塞、
允许动作和合同入口；S3 条目中的历史“下一步”不构成当前授权。

## FINAL_MAINLINE_CLOSEOUT：H-LOMA-CBOGT Scaled-Chen（2026-09-03）

### Final state

- Current phase: `MAINLINE CLOSEOUT`
- Scientific diagnosis: `closed`
- Repair: `confirmed`
- Integration: `qualified`
- F32: `qualified`
- Visualization: `completed`
- Runtime performance: `characterized`
- `mainline_state=closed_at_current_evidence_boundary`
- Current mainline next scientific task: `none`
- Future independent track: `REAL-HARDWARE DEPLOYMENT`; explicit future task name:
  `REAL_CMOS_RASPBERRY_PI_DEPLOYMENT`.

### Frozen authority and conclusion

The authority chain is `RMAL → S7 MPCDAR → S8 F64 R16-IQ → S9 F32-NQ → S9V full-path
visualization → S10 runtime characterization`. Later consumers use only their named immutable
parents; write-once earlier attempts remain historical and are never silently substituted.

Within the existing optical-path/CMOS evidence boundary, task-relevant information survives through
CMOS. The reconstruction failure is primarily supported as mapping/readout misalignment; simple
locality/raw Euclidean W-SVD is inadequate, useful mean/DC material is present, and centered
detector information is better represented by a fit-derived data-aware basis. Registered MPCDAR
rank 16 is the first confirmed candidate in its preregistered powers-of-two family, not a minimum, intrinsic, or
true rank-16 claim.

Frozen formula: `centered=z-mu_fit; h=centered@T16; prediction=c_mean+h@V16.T`. The S9 F32
artifact contract is `mu_fit[256]`, `c_mean[40000]`, `T16[256,16]`, `V16[40000,16]`, with
`684352` scalars, `2737408` raw tensor bytes, and `2739448` serialized bytes. The allowed system
description is `optical front end → CMOS intensity sensing → lightweight frozen electronic rank-16
readout`: hybrid optical-electronic reconstruction, not a native CMOS optical image.

S8 F64 is integration-qualified; S9 F32 is numerically qualified; S9V is
`descriptive_visualization_completed` and adds no scientific Gate; S10 is CPU-only readout
characterization, not measured speedup or production evidence. S10 recorded Python `3.11.16`,
PyTorch `2.5.1`, Windows `10.0.26100`, threads `8/8`, logical cores `16`, and CPU string
`AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD`; expected Windows `10.0.26200.0` did not
match. Batch mean latency B=`1/8/32` was `0.5601775/0.8914525/3.277882 ms`; mean throughput
was `1836.8588342/9338.7841702/9874.3015643 samples/s`; analytical MAC ratio was
`15.89825119236884`, not a measured speedup.

### Non-authorized boundary and future tracks

Native optical target-like CMOS reconstruction, real physical CMOS validation, Raspberry Pi
benchmarking, sim-to-real detector alignment, end-to-end hardware latency, GPU runtime, full-W
runtime, quantization, generative manifold, and generation readiness remain unestablished.
`reconstruction_manifold_supported=false`, `generative_manifold_supported=false`,
`generation_readiness=not_ready`, `training_authorized=false`, `formal_promotion=false`, and
`closed_task_relevant_bottleneck_not_supported` remain fixed. Future tracks are separate and require
new authority: real CMOS/Raspberry Pi deployment, sim-to-real alignment, optional optimization only
if separately justified, native optical redesign only with explicit authorization, and separate
generative-manifold/generation evidence. Older hypothesis entries remain for provenance and are not
deleted; their historical next-action text is superseded by this closeout.

Bridge closeout is `status=invalid`, `bridge_recovery_blocked=true`: the clean-HEAD `refresh`
stopped fail-closed on the existing invalid manifest
`C:\Users\24567\Documents\Codex\codex-original-chatgpt-bridge\current\.chatgpt-sources-manifest.json`.
The package was not deleted or overwritten, and no automatic upload occurred.

## 历史与稳定合同导航

Formal v2 identity、主线 closeout、旧研究路线与迁移状态的日期化正文统一见
[实验历史](experiment_history.md)、[v2 runtime](../architecture/v2_runtime.md) 和
[Windows-native migration](../migration_windows_native_20260830.md)。本交接不再复制旧实验正文、
完整 SHA 表或历史下一步；恢复资产仍按 `AGENTS.md` 与 outputs restore metadata 的不可变边界处理。

需要恢复某项历史研究时，先从实验历史定位 authority，再回到本文件顶部确认当前 branch、阻塞、
允许动作和是否需要新的独立授权。

</details>
