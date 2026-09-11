# 当前项目状态

更新时间：2026-09-11（LightGen 2D input/OLS alignment runtime 与 complex-OLS training consumer V2 已 IMPLEMENTED；corrected V2 source authority 已绑定；V2 long-training preflight 已 AUTHORIZABLE；未启动 300+300 scientific training；post-training control 未建立；bounded diagnosis VALID_COMPLETED，encoder/complex-OLS semantic collapse supported；证据截至本次本地核对）

## 当前主线，一页读完

**唯一活动研究主线：输入与 OLS 对齐 LightGen 的光学语义生成原型。**
目标和长期约束见 [项目记忆](PROJECT_MEMORY.md#唯一活动研究主线lightgen-输入与-ols-对齐的语义生成原型)。
`label + 8D` 是正式兼容基线，不是本轮研究的输入目标；N3 是已有组件研究，不等于论文复现完成。

| 问题 | 当前答案 |
| --- | --- |
| 已经有什么 | 真实复 OLS 直接解码证据；已冻结的 100 维 Gaussian 振幅 producer；其独立 sealed 确认通过 |
| 最新科学执行结果 | LightGen post-training `CONTROL_NOT_ESTABLISHED`；protected macro recall=`0.1617722231`，threshold=`0.937` |
| 最新合同状态 | V2 complex-OLS training consumer `IMPLEMENTED` 且已 rebind corrected contract；source-authority correction=`CONFIRMED`；V2 long-training preflight=`AUTHORIZABLE`；V2 CUDA scientific training 仍未授权 |
| 仍缺什么 | real-path positive control 尚未建立；300+300 scientific training、post-training evaluation、latent compatibility、direct prior generation、semantic distribution、diversity、novelty 均未完成；formal promotion/readiness 仍关闭 |
| 当前不能声称什么 | 光学语义生成成功、直接 prior generation 支持、严格复现论文、generation readiness 或 formal promotion |
| 既有执行合同的下一动作 | 仅在独立明确授权后执行 `EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V2`；不得由 AUTHORIZABLE preflight 自动启动训练或打开 sealed |
| 新结果对下一动作的约束 | corrected authority 与原合同科学语义一致，canonical source 未修改；preflight 只证明未来训练 harness 可执行，不能替代 scientific training authorization、generation acceptance 或 formal promotion |

## 最新 V2 long-training preflight：corrected authority 可执行（2026-09-11）

已将 V2 training consumer 与 preflight 的 active authority 从原 malformed contract
rebind 到 corrected contract。原合同 SHA-256=`b85fce30b70643638c1ea276d78dc58174c90980e420414cafc3f1947155de53` 保留为
historical-only；correction authority SHA-256=`fdf03a0a8568db947011549e7869ff11ab972e19c1e01fbc3e7d2e9d850dbeb0`，corrected
contract SHA-256=`aaa3cc409a9a4775be4bf48ef566fce8615f75b72277ca79194d36bd1f56c9cf`，canonical
`training.pt` SHA-256=`8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd`。两份合同的 scientific semantic normal form
相同，SHA-256=`cb90ac1bb33d56b155abf1cb2b01776983af356babbc45ccb8dded9ebd203203`；只发生 authority source SHA 修复、correction
lineage 和 active provenance binding 变化，未修改原合同或数据文件。

同一 `PREFLIGHT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_V2` 从头执行一次，最终结果为
`valid_completed/AUTHORIZABLE`，classification=`LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_V2_AUTHORIZABLE`。新的 immutable
input 为
[`lightgen_2d_semantic_complex_ols_long_training_preflight_v2_input.v1.json`](../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_long_training_preflight_v2_attempt_07/lightgen_2d_semantic_complex_ols_long_training_preflight_v2_input.v1.json)，SHA-256=`2e090926b057f70ed9cacc63473d16786336aa39b9371da2dcfdbccfead94b1d`；report
SHA-256=`b7bc2482d55af70ad8fc7a1b204c636425e9bca64448feda9e11e0111b5d64a4`；lineage SHA-256=`30913cd07ad318fbd94d2e5a4a346da6ca8390087e47a0b145d2c08b54b2d3bd`。

本次实际核验了目标 RTX 3060 Laptop GPU、compute capability `8.6`、PyTorch `2.5.1`、CUDA build `12.1`、VRAM
`6,441,926,656` bytes；Stage A/B 各完成 3-step CUDA engineering smoke，grid=`2048`、batch=`10`、Adam lr=`1e-3`。
55,000 TRAIN-only scale traversal 通过（649.11785 s，约 84.73 rows/s，float64 accumulation，epsilon=`1e-12`）；preflight scale
classification 被正式 loader 拒绝。normalized complex prior/mapping、A→B model carry-over 与 Adam reset、frozen scale、Stage-A/B
checkpoint/resume 均通过，resume 的 loss/parameter 最大差异均为 `0.0`；peak reserved VRAM=`4,592,762,880` bytes，minimum free
`557,125,632` bytes，storage feasible。semantic smoke recall 仅作 non-scientific firewall check，不作 selection。

数据 firewall 为 TRAIN=`55,000`、DEV semantic smoke=`1,000`、calibration=`0`、protected=`0`、test=`0`、sealed=`0`；
没有执行 300+300 scientific training、heldout evaluation、generation、formal promotion 或 readiness promotion。
`training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready` 保持。旧 blocked attempt-02 仍为
`BLOCKED_TRAINING_SOURCE_AUTHORITY_MISMATCH` 的 historical/superseded-by-authority-correction 证据；attempt-03 至 attempt-06 的 harness
修复过程也保留为不可覆盖的历史 artifacts。下一动作只有独立明确授权后的
`EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V2`，不得自动开始。

## 最新 V2 TRAIN source authority resolution：合同转录错误确认（2026-09-10）

对 `RESOLVE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_V2_TRAINING_SOURCE_AUTHORITY_V1` 完成一次只读 provenance
resolution。结果为 `VALID_COMPLETED`，classification=`TRAINING_SOURCE_AUTHORITY_CONTRACT_TRANSCRIPTION_ERROR_CONFIRMED`。
authority artifact 为
[`lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction.v1.json`](../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction_v1_attempt_01/lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction.v1.json)，
SHA-256=`fdf03a0a8568db947011549e7869ff11ab972e19c1e01fbc3e7d2e9d850dbeb0`；独立 corrected contract 为
[`lightgen_2d_semantic_complex_ols_training_contract.v2_corrected.v1.json`](../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction_v1_attempt_01/lightgen_2d_semantic_complex_ols_training_contract.v2_corrected.v1.json)，
SHA-256=`aaa3cc409a9a4775be4bf48ef566fce8615f75b72277ca79194d36bd1f56c9cf`。

原 V2 contract
[`lightgen_2d_semantic_complex_ols_training_contract.v2.json`](../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_training_contract_v2_attempt_01/lightgen_2d_semantic_complex_ols_training_contract.v2.json)
SHA-256=`b85fce30b70643638c1ea276d78dc58174c90980e420414cafc3f1947155de53` 的
`data_contract.source_artifacts.training.sha256` 实际为 63 位、正则无效的
`8151bb33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd`；当前
`data/MNIST/processed/training.pt` 未修改，完整 SHA-256 为
`8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd`，大小为 `47,521,438` bytes，
MNIST train schema 为 `60000×28×28 uint8` images + `60000 int64` labels。

N3 development source manifest（SHA=`4bc251af91b3a6cd0235fe113117443d871a564bd05634fa9e61b5ab0b36f812`）、
V1 training contract（`7d384fe9a73acc8f883badc4973926ab0e58bfce737067d4764deef10ee1d16f`）、V1 consumer input
（`c16dc2b5a1ed1c8a4f0ba0fc2e5a3e90154620e3f053d54d79342cc434e85aa7`）、V1 accepted preflight
（`365573edb0a3b4525bb571e31241bba6ba4243b9b2a3f591a8d1aea722c48d8a`）、V1 training report
（`f3f5fa876567513ec70cac67a6888aad6b55d140299992a535d0f9656bb4aa06`）、V1 training lineage
（`9c6eacb921b59788062236d9ba8354529f97bcf86de23813ba594e0fb7582c12`）和 post-training contract
（`b750bafd9c3866707a7d492c55c12ee08ab4775da9f6878805317feccb937a1d`）均对同一 canonical SHA
作 exact-source claim；没有发现其他 competing valid digest。因此结论是 source 文件保持 canonical，错误只发生在
V2 合同 authority 的转录。

corrected contract 保持原 schema/version/classification（`lightgen_2d_semantic_complex_ols_training_contract.v2` /
`2` / `DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONTRACT_V2_READY`），semantic normal form 与原合同相同，
SHA=`cb90ac1bb33d56b155abf1cb2b01776983af356babbc45ccb8dded9ebd203203`；除修正 source SHA 和新增
`authority_correction` lineage 元数据外，其余字段不变。原合同未覆盖，source 未修改，consumer rebind 必需。
本次 `CUDA=0`、optimizer steps=`0`、scientific training/generation=`0`，未读取 protected/test/sealed；
`training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready` 保持。

唯一下一动作是将 V2 consumer/preflight 显式 rebind 到上述 corrected contract，再执行新的
`PREFLIGHT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_V2` attempt；不得由 correction 自动推导训练授权。

## 最新实现结果：LightGen 2D semantic complex-OLS training consumer V2（2026-09-10）

按已批准的 V2 complex-OLS training contract 完成一次且仅一次 definition/implementation consumer，结果为
`valid_completed/IMPLEMENTED`，classification=`LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONSUMER_V2_IMPLEMENTED`。
合同 artifact 为
[`lightgen_2d_semantic_complex_ols_training_contract.v2.json`](../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_training_contract_v2_attempt_01/lightgen_2d_semantic_complex_ols_training_contract.v2.json)，
SHA-256=`b85fce30b70643638c1ea276d78dc58174c90980e420414cafc3f1947155de53`。

当前 consumer 为
[`lightgen_2d_semantic_complex_ols_training_v2.py`](../src/ldm_project/v2/application/research/lightgen_2d_semantic_complex_ols_training_v2.py)，
SHA-256=`d8b80121faf9da618a1710644b28fbf22755615aafba4d92cb0facb777f5fe97`；实现报告为
[`lightgen_2d_semantic_complex_ols_training_consumer_v2_implementation_report.v1.json`](../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_training_consumer_v2_attempt_02/lightgen_2d_semantic_complex_ols_training_consumer_v2_implementation_report.v1.json)，
SHA-256=`8c6a07e9abf302a2060b4e65b91b65d5c514cdf1257576eba5936f734e28ebbe`；long-training preflight input 为
[`lightgen_2d_semantic_complex_ols_long_training_preflight_v2.v1.json`](../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_training_consumer_v2_attempt_02/lightgen_2d_semantic_complex_ols_long_training_preflight_v2.v1.json)，
SHA-256=`93773f3f88daf8be73ba576e6264acfd093e02eb77af877818e6798d0b6a588c`。

实现冻结了 Stage A/B 各 300 steps、100/200/300 checkpoint cadence、`.937` semantic firewall、
train-only float64 per-port physical scale、normalized full-complex gauge-invariant MMD、精确 `CN(0,I_100)` prior、
Stage-A→B optimizer reset、状态机与 V2 checkpoint/resume identity。real encoded OLS 与 future sampled OLS
仍通过同一 `forward_from_ols`/`W_OLS`/generator 接口；没有引入 q-only latent 或 amplitude-first phase repair。

本次只执行了 CPU `NON_SCIENTIFIC_SMOKE`（128-grid、Stage A/B 各 1 update、synthetic scale/prior/checkpoint 测试）：
scientific steps=`0`，real data rows=`0`，protected/test rows=`0`，sealed rows=`0`。未执行 CUDA、真实训练、
heldout evaluation、generation 或 promotion；`training_authorized=false`、`formal_promotion=false`、
`generation_readiness=not_ready` 保持。该实现阶段的下一动作曾是独立的
`PREFLIGHT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_V2`；该 preflight 已在新 attempt-02
因 TRAIN source authority mismatch 有效阻断，不得由本实现自动启动训练；该 mismatch 后续已由独立 provenance
resolution 确认为合同字段转录错误，当前 consumer 仍待 rebind 到 corrected contract。

## 最新 V2 long-training preflight 结果：TRAIN source authority 阻断（2026-09-10）

按 `PREFLIGHT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_V2` 的冻结输入执行新建
`attempt-02`。结果为 `valid_failed/BLOCKED`，classification=
`BLOCKED_TRAINING_SOURCE_AUTHORITY_MISMATCH`。报告为
[`lightgen_2d_semantic_complex_ols_long_training_preflight_v2_report.v1.json`](../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_long_training_preflight_v2_attempt_02/lightgen_2d_semantic_complex_ols_long_training_preflight_v2_report.v1.json)，
SHA-256=`2fbea9060fc96a2634201366dc2cb5bb549491581cc4fca3aa518a033750f7b3`；lineage 为
[`lightgen_2d_semantic_complex_ols_long_training_preflight_v2_lineage.v1.json`](../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_long_training_preflight_v2_attempt_02/lightgen_2d_semantic_complex_ols_long_training_preflight_v2_lineage.v1.json)，
SHA-256=`1f9abc34454ef26734081c2b5c86352d624970b359558ccc8542ce2f71ddae74`。

冻结 V2 contract（SHA-256=`b85fce30b70643638c1ea276d78dc58174c90980e420414cafc3f1947155de53`）要求
`data/MNIST/processed/training.pt` 为
`8151bb33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd`，但当前本地文件的完整 SHA-256
为 `8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd`。合同、consumer 与 implementation
preflight input 彼此一致，冲突发生在冻结 authority 与实际输入文件之间；因此不能把实际值静默写回合同，亦未修改
合同或数据文件。

该硬停止发生在加载训练张量前：未进入目标 CUDA/backend 检查、Stage A/B smoke、DEV semantic firewall、
55,000-row scale extraction、checkpoint/resume 或任何 scientific training；未读取 protected/test/sealed，
没有 preflight scale 或 checkpoint 生成。`training_authorized=false`、`formal_promotion=false`、
`generation_readiness=not_ready` 保持。该 authority gate 已由独立 correction artifact 解析完成；下一步仅为
consumer rebind 后用新 attempt 重走完整 preflight，不得绕过该 gate 直接训练。

### 证据入口与解释边界

- [真实复 OLS 直接重放](../outputs/20260906_v2/runs/lightgen_scaled_chen_direct_ols_semantic_decode_canary_v1_attempt_01/compact_report.json)：
  selection/confirmation 语义准确率 0.973/0.9754，与 full path 一致。是 observed-source replay，
  不是新样本生成或人眼视觉质量验收。
- [N3 独立确认](../outputs/20260910_v2/runs/n3_selected_producer_sealed_confirmation_v1_attempt_01/n3_selected_producer_sealed_confirmation_lineage.v1.json)：
  只支持冻结振幅 producer 的该次确认。sealed 已消费，不可 reset/retry 或复用为新独立确认。
- [完整复 OLS 合同](../outputs/20260910_v2/runs/n3_complex_ols_amplitude_phase_joint_compatibility_contract_v1_attempt_01/n3_complex_ols_amplitude_phase_joint_compatibility_contract.v1.json)：
  冻结了 Gaussian 振幅与独立抽取的真实相对相位向量组合的诊断假设；合同执行结果见下方 report/lineage。
- [完整复 OLS 联合兼容性 report](../outputs/20260910_v2/runs/n3_complex_ols_amplitude_phase_joint_compatibility_v1_attempt_01/n3_complex_ols_amplitude_phase_joint_compatibility_report.v1.json)：
  control=`5/5`，唯一 candidate=`0/5`，classification=`N3_COMPLEX_OLS_INDEPENDENT_PHASE_INHERITANCE_NOT_SUPPORTED`。
- [完整复 OLS 联合兼容性 lineage](../outputs/20260910_v2/runs/n3_complex_ols_amplitude_phase_joint_compatibility_v1_attempt_01/n3_complex_ols_amplitude_phase_joint_compatibility_lineage.v1.json)：
  记录 V2 exact 5000-row complex materialization、control/candidate artifact 与治理边界。
- [q-conditioned phase coupling contract](../outputs/20260910_v2/runs/n3_minimal_q_conditioned_phase_coupling_contract_v1_attempt_01/n3_minimal_q_conditioned_phase_coupling_contract.v1.json)：
  SHA-256=`14f4f3cd5d1046eb648b3410593d778910f8b8d33262528f9e0f12ba94250434`；冻结唯一
  candidate=`Q_CONDITIONED_NEAREST_PHASE_DONOR_V1`、`K=1` 及未来 evaluator，已按合同执行一次。
- [q-conditioned nearest phase-donor report](../outputs/20260910_v2/runs/n3_q_conditioned_nearest_phase_donor_v1_attempt_01/n3_q_conditioned_nearest_phase_donor_report.v1.json)：
  SHA-256=`d17a472558349b043da07bc51310e7a350263608c059f37361237b1000318bcb`；status/result=
  `valid_completed/VALID_COMPLETED`，classification=`N3_COMPLEX_OLS_Q_CONDITIONED_NEAREST_PHASE_DONOR_NOT_SUPPORTED`，
  candidate=`0/5`，五折 `p_MC=0.001` 且每折 exceedance=`0`。
- [q-conditioned nearest phase-donor lineage](../outputs/20260910_v2/runs/n3_q_conditioned_nearest_phase_donor_v1_attempt_01/n3_q_conditioned_nearest_phase_donor_lineage.v1.json)：
  SHA-256=`735fc652d334b2c4cc2a7b70a57c100d242149a17f72bb9d82a2de5b92f79c67`；绑定 immutable candidate
  banks/calibration artifact，记录 fitting-only donor pool、完整 100-D phase inheritance 与治理边界。
- [higher-expressivity q-conditioned phase contract](../outputs/20260910_v2/runs/n3_q_conditioned_complex_phasor_linear_residual_bootstrap_contract_v1_attempt_01/n3_q_conditioned_complex_phasor_linear_residual_bootstrap_contract.v1.json)：
  SHA-256=`8f19c3dbb6b9bc17560276e7f1cf61004fd4b0583a3c8a15792b4a86fa05c29f`；status/result=
  `valid_completed/READY`，唯一 candidate=`Q_CONDITIONED_COMPLEX_PHASOR_LINEAR_RESIDUAL_BOOTSTRAP_V1`。
- [higher-expressivity q-conditioned phase report](../outputs/20260910_v2/runs/n3_q_conditioned_complex_phasor_linear_residual_bootstrap_v1_attempt_01/n3_q_conditioned_complex_phasor_linear_residual_bootstrap_report.v1.json)：
  SHA-256=`ae60752a27d3fb655b9354ace5591cb7f092db2aebb6206774853fa82cd025fe`；status/result=
  `valid_completed/VALID_COMPLETED`，classification=`N3_COMPLEX_OLS_Q_CONDITIONED_COMPLEX_PHASOR_LINEAR_RESIDUAL_NOT_SUPPORTED`，
  candidate=`0/5`，五折均 `p_MC=0.001` 且 exceedance=`0`。
- [higher-expressivity q-conditioned phase lineage](../outputs/20260910_v2/runs/n3_q_conditioned_complex_phasor_linear_residual_bootstrap_v1_attempt_01/n3_q_conditioned_complex_phasor_linear_residual_bootstrap_lineage.v1.json)：
  SHA-256=`503c5afa463403d237768e5eee1c5e130939394486e1c8348716ef553041a9de`；绑定 control=`5/5`
  的复用、五折 candidate diagnostics 与治理边界。
- [higher-expressivity candidate artifact](../outputs/20260910_v2/runs/n3_q_conditioned_complex_phasor_linear_residual_bootstrap_v1_attempt_01/n3_q_conditioned_complex_phasor_linear_residual_bootstrap_banks_and_calibration.v1.pt)：
  SHA-256=`e944a96e4f751db35ec84440b779cb7a8825145a78ea725703f839e3c4f9c587`；包含 5×4×200
  complex model banks、5×999 calibration trials 与 fold-local phase coefficients。

### 执行与文档边界

活动分支为 `research/lightgen-ols-native-semantic-generation-v1`，执行前后 HEAD 均为
`a5c07512614944d4be6392ae29d48e0b189bea4b`，工作树已有未提交研究修改。机器
[governance state](status/research_governance_state.json) 保持不变，当前
`active_objective=N3_COMPLEX_OLS_INDEPENDENT_PHASE_INHERITANCE_NOT_SUPPORTED`，
`hard_blocker=N3_COMPLEX_OLS_MINIMAL_Q_CONDITIONED_PHASE_COUPLING_CONTRACT_REQUIRED`。
本次已按 READY 合同唯一授权动作
`EXECUTE_N3_Q_CONDITIONED_COMPLEX_PHASOR_LINEAR_RESIDUAL_BOOTSTRAP_V1` 完成一次 CPU-only formal
diagnostic execution；结果为 candidate=`0/5`、classification=`N3_COMPLEX_OLS_Q_CONDITIONED_COMPLEX_PHASOR_LINEAR_RESIDUAL_NOT_SUPPORTED`。
bounded architectural decision 已完成：采用 `LIGHTGEN_INPUT_OLS_ALIGNMENT_FIRST`，并发布
[`lightgen_2d_semantic_generation_input_ols_alignment_contract.v1.json`](../outputs/20260910_v2/runs/lightgen_2d_semantic_generation_input_ols_alignment_contract_v1_attempt_01/lightgen_2d_semantic_generation_input_ols_alignment_contract.v1.json)，
SHA-256=`6f3859d87e9e3f38d141943ae3fac23607cc446fe2c58b99c9b5b4c379a3a72e`，
status/result=`valid_completed/READY`。该契约只冻结 2D 语义图像生成的输入、10×10/100-port
complex OLS、强制 modal projection/re-emission 与共用 `W_OLS` generator-input writer；它不是论文
完整复现、训练授权或物理 generation 结论。

`training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready` 保持。
本次执行加载了已物化且 hash-bound 的 V2 q/complex development rows，使用同折 fitting-only Gaussianized-q
输入、固定 101 列 OLS、完整 100-D complex residual bootstrap、unit-phasor projection 与现有 gauge；没有加载
sealed values，历史 full-complex control=`5/5` 仅复用未重跑。未使用 CUDA、optimizer、backward、training、
generator、image evaluation 或 optical forward；selected producer 未修改，machine governance SHA 保持
`3d5b55bbcb3bb01cffb7badefb2a1ea2fd7cf9e924afa17729b183d0dff42e99`。
本次 definition-only 对纸面和代码 authority 做了只读核对，未加载 q/complex/sealed 数值；后续交接见 [current_handoff](status/current_handoff.md)。本地文档已更新，Sources 未 refresh/上传，
不能声称云端已同步或 publication 为 current_clean。

## LightGen 2D semantic complex-OLS post-training evaluation contract（2026-09-10）

完成一次且仅一次 definition-only consumer。契约为
[`lightgen_2d_semantic_complex_ols_post_training_evaluation_contract.v1.json`](../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_post_training_evaluation_contract_v1_attempt_01/lightgen_2d_semantic_complex_ols_post_training_evaluation_contract.v1.json)，
SHA-256=`b750bafd9c3866707a7d492c55c12ee08ab4775da9f6878805317feccb937a1d`，
status/result=`valid_completed/READY`，classification=`DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_POST_TRAINING_EVALUATION_CONTRACT_V1_READY`。

契约只绑定正式 training report 选出的 step-100 checkpoint（SHA=`6cfcc82831b575d83c097648a78b6507b32900e578d00c446d76ccd29130e149`），
以及独立 acceptance-tier `FrozenMNISTEvaluator v1`（SHA=`46741ec76e45653c9dcb76e7b4f1645c9c057a47cdd9e4c1778d9debb553ec57`，validation accuracy=`0.9875999689102173`）。
raw CMOS 到 evaluator 的唯一视图是 clamp `[0,1]` 后 adaptive average pool 到 `28×28`；这是 PROJECT_DECISION，
不宣称为 Chen 事实，也不宣称 output-domain requalification。此前 validation=`0.9712` 的 output-domain evaluator 被显式拒绝。

未来一次评估固定为：real image positive control → full-complex `z∈C^100` latent compatibility → 四个 `P_target` 直接生成 bank →
同一 `W_OLS`/同一 generator 的 semantic feature distribution、diversity 与 novelty。development selection 之外的 4000 行
（400/class，index SHA=`c3db6fe1554baec2bebed85cdbc03cfd13f74521fd658244e6d2ef9aabf52ee9`）只作 control threshold 与 real-data null calibration；
official test `10000` 行只绑定 SHA，未由本次定义读取。B=`999`、alpha=`0.01`、无 FID/KID、无重试/筛选/回退，且 sealed 永不重用。

本次未执行 CUDA、training、protected/heldout scientific evaluation、MMD 或 generation；
`training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready` 保持。下一动作仅为取得独立明确授权后执行
`EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_POST_TRAINING_EVALUATION_V1`。

## 当前 post-training evaluation 结果：LightGen 2D semantic complex-OLS（2026-09-10）

按用户对唯一 scope `EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_POST_TRAINING_EVALUATION_V1` 的明确授权，
消费未修改的 evaluation contract（SHA-256=`b750bafd9c3866707a7d492c55c12ee08ab4775da9f6878805317feccb937a1d`）
和唯一 step-100 checkpoint（SHA-256=`6cfcc82831b575d83c097648a78b6507b32900e578d00c446d76ccd29130e149`），
完成一次且仅一次 Windows-native CUDA protected evaluation。结果为 `VALID_COMPLETED`，最终 classification 为
`LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_POST_TRAINING_CONTROL_NOT_ESTABLISHED`。

正式 report 为
[`lightgen_2d_semantic_complex_ols_post_training_evaluation_report.v1.json`](../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_post_training_evaluation_v1_attempt_01/lightgen_2d_semantic_complex_ols_post_training_evaluation_report.v1.json)，
SHA-256=`4048de57a35a1899240a231cfbeec70ed5950275b2014fbcb1dfc96c664c4996`；lineage 为
[`lightgen_2d_semantic_complex_ols_post_training_evaluation_lineage.v1.json`](../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_post_training_evaluation_v1_attempt_01/lightgen_2d_semantic_complex_ols_post_training_evaluation_lineage.v1.json)，
SHA-256=`9f1a89bafa8f57f2d6cd4242de2153a2cd5d4c02ac2d2ad82a652680a204f19a`；control artifact SHA-256=
`c5b2cf0bd2d98b5eceb4a53881312c1397bda91e68a87def788364d72244a7a3`。

Stage 1 real-image positive control 使用完整 protected test role（10000 rows）并走完
`image → aligned SLM/relay → encoder → modal projection z_real∈C^100 → T=I_100 → W_OLS → same generator → raw CMOS`。
clean 4000-row calibration macro recall=`0.987`，冻结 threshold=`0.937`；protected real-path macro recall 仅为
`0.1617722231`（per-class correct=`[5,0,396,451,7,311,122,18,202,75]`），故 control FAIL。

按合同 fail-closed 停止：没有执行 latent compatibility、4×250 direct `P_target` generation、semantic distribution、
diversity 或 novelty；没有生成 artifact，没有重训、checkpoint search、threshold/evaluator/prior 调整，也没有加载或复用
sealed 数值。protected source 值只读取一次；branch 与 HEAD 前后均为
`research/lightgen-ols-native-semantic-generation-v1` / `a5c07512614944d4be6392ae29d48e0b189bea4b`。
`training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready` 保持。下一步不是自动重训或重跑，
而是先对 real-path control failure 做独立的 bounded diagnosis/planning。

## 当前 real-path control failure diagnosis 结果：LightGen 2D semantic complex-OLS（2026-09-10）

按用户明确授权完成一次且仅一次
`DIAGNOSE_LIGHTGEN_2D_SEMANTIC_REAL_PATH_CONTROL_FAILURE_V1`。结果为 `VALID_COMPLETED`，primary diagnosis 为
`REAL_PATH_FAILURE_ENCODER_OLS_SEMANTIC_COLLAPSE_SUPPORTED`。

诊断只使用 train、checkpoint-selection development、calibration 三个非 protected 角色，并只读已发布的
Stage-1 protected aggregate；没有重新 forward protected/test，没有 training、optimizer step、checkpoint search、
evaluator/threshold 调整、prior-to-generator generation 或 sealed reuse。三类 real-path macro recall 为
`train=0.158`、`development=0.170`、`calibration=0.16775`，而原始图像分别为 `0.998/0.990/0.987`。

complex OLS 的 calibration mean power=`2.2122145782e-10`，冻结 prior mean power=`1.0001851933e-6`，
ratio=`2.2118049667e-4`；gauge-invariant effective rank=`16.4476411`，between/within class-centroid
distance ratio=`0.9402647`，full-complex MMD=`0.0512237255`。raw CMOS 仍有非零空间与样本间离散，
same-`W_OLS`/generator 路径误差为 0；encoder image-vs-latent gradient cosine=`0.0349641`，latent/image
weighted norm ratio=`0.0257622`，不支持 active latent regularization conflict。

诊断 report 为
[`lightgen_2d_semantic_real_path_control_failure_diagnosis_report.v1.json`](../outputs/20260910_v2/runs/lightgen_2d_semantic_real_path_control_failure_diagnosis_v1_attempt_01/lightgen_2d_semantic_real_path_control_failure_diagnosis_report.v1.json)，
SHA-256=`8b98dbcb2955417ca4dacd2298144bfdc006fd697dc1dc36eb49ff63dadcda86`；lineage 为
[`lightgen_2d_semantic_real_path_control_failure_diagnosis_lineage.v1.json`](../outputs/20260910_v2/runs/lightgen_2d_semantic_real_path_control_failure_diagnosis_v1_attempt_01/lightgen_2d_semantic_real_path_control_failure_diagnosis_lineage.v1.json)，
SHA-256=`ce3f8a5101a300a764d4d005f9ca2eecfccc4db3c4a1a2dd1ca3860e511ba914`。

secondary findings：evaluator/domain mismatch、数值/图像 collapse、latent-vs-image conflict、optical-vs-image
conflict 和 train/dev/protected shift 均不支持；generator-side decode、checkpoint objective mismatch 与
insufficient bounded progress 保持 unresolved。该 diagnosis 不证明 LightGen 对齐架构、complex Gaussian prior、
gauge-invariant MMD 或 100-port OLS 无效。下一步只能停止在 diagnosis，若继续须另行定义一个针对 encoder/OLS
semantic collapse 的修正 training contract；不自动训练。

## LightGen 2D semantic complex-OLS training contract（2026-09-10）

在 aligned input/OLS runtime 已 IMPLEMENTED 且结构性验证通过后，完成一次且仅一次
definition-only consumer。契约为
[`lightgen_2d_semantic_complex_ols_training_contract.v1.json`](../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_training_contract_v1_attempt_01/lightgen_2d_semantic_complex_ols_training_contract.v1.json)，
SHA-256=`7d384fe9a73acc8f883badc4973926ab0e58bfce737067d4764deef10ee1d16f`，
status/result=`valid_completed/READY`，classification=`DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONTRACT_V1_READY`。

契约冻结 future training 的接口边界：唯一权威 latent 为完整 `z∈C^100`；real encoded OLS 与 future
generated OLS 都经过同一强制 modal projection、coefficient-space `T=I_100`、`W_OLS` re-emission
和同一 generator。可训练面仅为 aligned runtime 的 `encoder.phase` 与 `generator.phase`；OLS、relay、
modal basis、transmission、normalization 和 evaluator 全部冻结。未来目标分布为 native complex isotropic
Gaussian，latent loss 为固定 bandwidth 的 full-complex gauge-invariant MMD，图像项为 raw CMOS intensity
MSE；q、q→phase reconstruction、历史 phase producer、电子 decoder 和 sealed 数值均不进入主线。

该 READY 本身只表示训练契约可定义、可被后续 consumer 实现，并不自动授权 CUDA、optimizer、training、heldout
scientific evaluation、generation acceptance、formal promotion 或 generation readiness。随后用户对唯一 scope
`EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V1` 作出明确授权，且该次 frozen 100-step CUDA training
已完成；definition artifact 中的 `training_authorized=false` 未被改写，`formal_promotion=false`、
`generation_readiness=not_ready` 仍保持。

## LightGen 2D semantic complex-OLS training consumer（2026-09-10）

按上述 hash-bound training contract 完成一次且仅一次实现与 preflight。consumer 位于
[`lightgen_2d_semantic_complex_ols_training.py`](../src/ldm_project/v2/application/research/lightgen_2d_semantic_complex_ols_training.py)，
入口为
[`execute_lightgen_2d_semantic_complex_ols_training_v1.py`](../scripts/research/execute_lightgen_2d_semantic_complex_ols_training_v1.py)，
结果为 `valid_completed/IMPLEMENTED`，classification=`LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONSUMER_V1_IMPLEMENTED`。

preflight artifact 为
[`lightgen_2d_semantic_complex_ols_training_preflight_input.v1.json`](../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_training_consumer_v1_attempt_01/lightgen_2d_semantic_complex_ols_training_preflight_input.v1.json)，
SHA-256=`c16dc2b5a1ed1c8a4f0ba0fc2e5a3e90154620e3f053d54d79342cc434e85aa7`。
CPU engineering smoke 使用真实 train-role 数据完成 3 个更新；仅 `encoder.phase` 和 `generator.phase`
（共 40,000 个 float32 参数）可训练，modal projection 与 `W_OLS` 均被证明位于实际 autograd path。
full-complex prior、gauge-invariant MMD、raw CMOS MSE、固定权重 total loss、Adam manifest、参数/冻结状态、
safe checkpoint reload 和 resume continuation 均通过；checkpoint 为 smoke-only，SHA-256=
`6df6254196e24dec1cd7f902ca178de07efa022634c380326765a29250bce7be`，不可晋升。

该 consumer/preflight 阶段未执行 100-step future training、heldout evaluation、generation acceptance、formal
promotion 或 generation readiness；official MNIST test split 未读取，consumed N3 sealed evidence 未读取/复用。
随后正式训练已在独立明确授权下完成，详见下节；该 smoke checkpoint 仍不可晋升。

## LightGen 2D semantic complex-OLS long-training preflight（2026-09-10）

按 READY training contract 完成一次且仅一次
`PREFLIGHT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_V1`，结果为
`valid_completed/AUTHORIZABLE`，classification=`LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_AUTHORIZABLE`。
唯一报告为
[`lightgen_2d_semantic_complex_ols_long_training_preflight.v1.json`](../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_long_training_preflight_v1_attempt_01/lightgen_2d_semantic_complex_ols_long_training_preflight.v1.json)，
SHA-256=`365573edb0a3b4525bb571e31241bba6ba4243b9b2a3f591a8d1aea722c48d8a`；它绑定 training contract
SHA=`7d384fe9a73acc8f883badc4973926ab0e58bfce737067d4764deef10ee1d16f`、consumer input SHA=
`c16dc2b5a1ed1c8a4f0ba0fc2e5a3e90154620e3f053d54d79342cc434e85aa7` 及当前 consumer、entrypoint、aligned runtime
身份。

目标设备为 `NVIDIA GeForce RTX 3060 Laptop GPU`，compute capability=`8.6`，单 GPU，VRAM=
`6441926656` bytes；PyTorch=`2.5.1`、CUDA build=`12.1`，complex exp/FFT/polar 与 NVRTC backend probes
均通过。production propagation grid=`2048`，exact logical batch=`10`；同一 production consumer 完成 3 个真实
CUDA forward/loss/backward/optimizer-step updates，loss、complex prior、raw CMOS、梯度与参数均 finite，modal
projection 与 `W_OLS` active，仅 `encoder.phase`/`generator.phase` 可更新，冻结状态未变。

preflight 记录 forward peak allocated=`2753601536` bytes、backward peak allocated=`2461667840` bytes、
minimum observed free=`2436896624` bytes；target-device smoke checkpoint 为
[`lightgen_2d_semantic_complex_ols_long_training_preflight_smoke_checkpoint.v1.pt`](../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_long_training_preflight_v1_attempt_01/lightgen_2d_semantic_complex_ols_long_training_preflight_smoke_checkpoint.v1.pt)，
SHA-256=`2ad10fd883cd4044627a7ae0d4ddc42754e3510e9dc13c78edfcb2ef6487d8aa`，明确
`smoke_only=true`、`accepted_training_checkpoint=false`。save→reload→continuation 的参数、optimizer、RNG、
prior RNG、sampler 与 contract/runtime identity 均一致，next-step loss/parameter difference 均为 `0.0`；非 identity
coefficient transport negative control 按预期被 `T=I_100` 防火墙拒绝。

仅使用 local MNIST train split 与 development selection（500/class、selection 1000、seed=`4321`）；official test
split 未读取，heldout/sealed 均为 false。该 preflight 本身不等于训练授权；随后正式训练已在独立明确授权下完成。
当前仍为 `formal_promotion=false`、`generation_readiness=not_ready`；训练合同中的
`training_authorized=false` 定义标志未被改写。

## LightGen 2D semantic complex-OLS formal training（2026-09-10）

按用户对唯一 scope `EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V1` 的明确授权，使用已通过的
target-device preflight、冻结 training contract 和当前 production consumer，完成一次且仅一次正式 100-step
CUDA training。结果为 `VALID_COMPLETED`，status=`valid_completed`，classification=
`LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V1_VALID_COMPLETED`；100/100 updates 完成，运行约 `67.36 s`。

正式 report 为
[`lightgen_2d_semantic_complex_ols_training_report.v1.json`](../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_training_v1_attempt_01/lightgen_2d_semantic_complex_ols_training_report.v1.json)，
SHA-256=`f3f5fa876567513ec70cac67a6888aad6b55d140299992a535d0f9656bb4aa06`；lineage 为
[`lightgen_2d_semantic_complex_ols_training_lineage.v1.json`](../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_training_v1_attempt_01/lightgen_2d_semantic_complex_ols_training_lineage.v1.json)，
SHA-256=`9c6eacb921b59788062236d9ba8354529f97bcf86de23813ba594e0fb7582c12`。目标设备为
`NVIDIA GeForce RTX 3060 Laptop GPU`、compute capability=`8.6`，production grid=`2048`，exact logical
batch=`10`；PyTorch=`2.5.1`、CUDA build=`12.1`。

只有 `encoder.phase`、`generator.phase` 可训练，共 `40,000` 个参数；full complex `z∈C^100`、mandatory
modal projection、`T=I_100`、`W_OLS`、full-complex gauge-invariant MMD 与 raw CMOS MSE 均按冻结合同生效。
所有训练 trace finite，冻结状态未变；没有 q-only latent、phase repair 或历史 producer 进入该运行。初始 train
`L_total=0.18129613995552063`，最终为 `0.15601420402526855`；step-50 dev
`L_total=0.12443317954406738`，step-100 dev `L_total=0.12301895964050293`，按固定规则选择 step 100。

step-50 checkpoint 为
[`lightgen_2d_semantic_complex_ols_training_checkpoint_step_050.v1.pt`](../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_training_v1_attempt_01/lightgen_2d_semantic_complex_ols_training_checkpoint_step_050.v1.pt)，
SHA-256=`c467749eba3199fa4bc3815c76119a68571cac20167ea05cc735b1b04b1c23ff`；step-100 checkpoint 为
[`lightgen_2d_semantic_complex_ols_training_checkpoint_step_100.v1.pt`](../outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_training_v1_attempt_01/lightgen_2d_semantic_complex_ols_training_checkpoint_step_100.v1.pt)，
SHA-256=`6cfcc82831b575d83c097648a78b6507b32900e578d00c446d76ccd29130e149`，为 fixed dev selection 的
selected training checkpoint，但尚未被接受为 formal/generation checkpoint。

数据仅使用 frozen train/dev roles；official test、heldout、sealed 均未读取或复用，post-training evaluation 和
scientific generation 也未执行。`formal_promotion=false`、`generation_readiness=not_ready` 保持；下一步仅定义
`DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_POST_TRAINING_EVALUATION_CONTRACT_V1`，不自动评估或生成。

## LightGen 2D semantic-generation input/OLS alignment contract（2026-09-10）

bounded architectural decision 的结果为 `READY`：`LIGHTGEN_INPUT_OLS_ALIGNMENT_FIRST`。契约明确：

`X → explicit SLM amplitude mapping → illuminated complex field → coherent input relay → optical encoder → z_real∈C^100 → W_OLS(z_real) → photonic generator → raw CMOS intensity`

以及收敛的生成路径：

`z_generated∈C^100 → W_OLS(z_generated) → same photonic generator → raw CMOS intensity`。

权威 OLS 状态是保持相对相位的 `z∈C^100`，端口固定为 row-major `j=10*r+c` 的 10×10 单模接口；
`z_j=<E_encoder,u_j>`，先做系数空间 `T=I_100`，再用
`W_OLS(z)=sum_j(Tz)_j v_j` 强制 modal re-emission。`q` 仅为派生 diagnostic/statistic，不能替代
complex latent；整幅场 identity bypass、电子 decoder 和 amplitude-first phase repair 不属于对齐主线。

论文事实、论文未披露细节和项目决策已在契约内分列；其中 `sqrt(image)`、scaled LP01 参数、
row-major port order、`T=I_100` 和当前 2+2 diffractive baseline 都明确标为 `PROJECT_DECISION`，
而非 Chen fact。该契约随后已由 runtime 实现消费者执行；其 long-training preflight 与正式 training 均已完成；当前下一动作是
`DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_POST_TRAINING_EVALUATION_CONTRACT_V1`；
本契约不授权 CUDA、training、heldout scientific evaluation、generation acceptance、formal promotion
或 sealed reuse。

## LightGen 2D semantic-generation input/OLS alignment runtime（2026-09-10）

按上述契约（SHA-256=`6f3859d87e9e3f38d141943ae3fac23607cc446fe2c58b99c9b5b4c379a3a72e`）完成一次实现与结构性验证，结果为
`LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_RUNTIME_V1_IMPLEMENTED`。runtime 位于
[`lightgen_2d_semantic_generation_input_ols_alignment.py`](../src/ldm_project/v2/application/research/lightgen_2d_semantic_generation_input_ols_alignment.py)，
SHA-256=`d01a234f6486a99c7ac2307b1b4021ed2ffccb53812a0a858675617a9023c3ce`。

实现冻结两条共用路径：`forward_from_image(X)` 显式产生 SLM intensity/amplitude、零相位 illuminated
field、输入 relay、2-layer phase-only encoder、`z_real∈C^100`；`forward_from_ols(z)` 只接受 canonical
complex OLS。两者均调用唯一 `w_ols` → `T=I_100` coefficient transmission → scaled-LP01 modal
re-emission → 同一 `generator`（2-layer phase-only）→ B=0 output relay → raw CMOS `|E_out|²`。
不存在 label+8D、q-only、phase-repair、电子 decoder 或 whole-field identity bypass 依赖。

OLS overlap primitives 在 [`fiber_operator.py`](../src/ldm_project/v2/optics/fiber_operator.py) 中做了最小因子化，
保留旧 `FiberArrayOLS.forward` 行为并公开可微的 projection/transmission/re-emission 边界；其最终 SHA-256 为
`6e941d6610c282aeb9f54cb39ea200c868f7b0eb14c3bc41b692ee505f4cf295`。独立验证消费者及报告为
[`verify_lightgen_2d_semantic_generation_input_ols_alignment_runtime_v1.py`](../scripts/research/verify_lightgen_2d_semantic_generation_input_ols_alignment_runtime_v1.py)
和 [`lightgen_2d_semantic_generation_input_ols_alignment_runtime_report.v1.json`](../outputs/20260910_v2/runs/lightgen_2d_semantic_generation_input_ols_alignment_runtime_v1_attempt_02/lightgen_2d_semantic_generation_input_ols_alignment_runtime_report.v1.json)，
报告 SHA-256=`e26d24654e1d9c8707e9a6b93b5c5ee66f1172fb0ae7cd686d24ea03c9263af4`。

报告确认 complex shape/dtype=`[1,100]`/`torch.complex64`、row-major order、active modal projection、
`T=I_100`、generator-input plane compatibility、same-W_OLS path equivalence、same-generator raw equivalence、
direct path skips image encoder 和 q-only rejection。验证使用 CPU、显式 128 propagation grid 仅作结构 smoke；生产默认仍为
合同项目基线 2048。全程未执行 CUDA、optimizer、training、heldout scientific evaluation、generation acceptance、
FID/classifier、formal promotion 或 sealed reuse；`training_authorized=false`、`formal_promotion=false`、
`generation_readiness=not_ready` 保持。complex-OLS training contract、consumer、long-training preflight 与正式
training 已分别为 READY/IMPLEMENTED/AUTHORIZABLE/VALID_COMPLETED；下一步只定义 post-training evaluation
contract，不自动评估或生成。

## N3 higher-expressivity q-conditioned complex-phasor residual-bootstrap execution（2026-09-10）

按 READY contract 的唯一授权动作完成一次且仅一次 CPU-only execution。冻结 contract SHA 为
`8f19c3dbb6b9bc17560276e7f1cf61004fd4b0583a3c8a15792b4a86fa05c29f`；report 为
[`n3_q_conditioned_complex_phasor_linear_residual_bootstrap_report.v1.json`](../outputs/20260910_v2/runs/n3_q_conditioned_complex_phasor_linear_residual_bootstrap_v1_attempt_01/n3_q_conditioned_complex_phasor_linear_residual_bootstrap_report.v1.json)，
SHA=`ae60752a27d3fb655b9354ace5591cb7f092db2aebb6206774853fa82cd025fe`；lineage 为
[`n3_q_conditioned_complex_phasor_linear_residual_bootstrap_lineage.v1.json`](../outputs/20260910_v2/runs/n3_q_conditioned_complex_phasor_linear_residual_bootstrap_v1_attempt_01/n3_q_conditioned_complex_phasor_linear_residual_bootstrap_lineage.v1.json)，
SHA=`503c5afa463403d237768e5eee1c5e130939394486e1c8348716ef553041a9de`。

唯一 candidate `Q_CONDITIONED_COMPLEX_PHASOR_LINEAR_RESIDUAL_BOOTSTRAP_V1` 的 phase OLS 每折均为
rank=`101/101`，condition number 约 `22.28–22.62`，projection invalid coordinates=`0`；但在冻结
`energy_distance` primary、`B=999`、`p_MC=(1+count(T_b>=T_obs))/(B+1)` 与 strict `5/5` 下，五折
均 exceedance=`0`、`p_MC=0.001`，所以 candidate=`0/5`，分类为
`N3_COMPLEX_OLS_Q_CONDITIONED_COMPLEX_PHASOR_LINEAR_RESIDUAL_NOT_SUPPORTED`：

| fold | rank | T_candidate | exceedance | p_MC | pass |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 0 | 101/101 | 0.2222192429 | 0 | 0.001 | false |
| 1 | 101/101 | 0.2399564571 | 0 | 0.001 | false |
| 2 | 101/101 | 0.2419473636 | 0 | 0.001 | false |
| 3 | 101/101 | 0.2473423824 | 0 | 0.001 | false |
| 4 | 101/101 | 0.2703857960 | 0 | 0.001 | false |

历史 `FULL_COMPLEX_ROW_EMPIRICAL_BOOTSTRAP_CONTROL=5/5` 复用且未重跑；sealed confirmation
`consumed=true` 仅作为既有治理 provenance，`values_loaded=false`、`reused=false`。本次不支持的是该
固定 complex-phasor linear conditional-mean + complete-vector residual 假设，不是对所有 conditional
phase models、LightGen 机制、论文复现或物理 generation 的普遍否定。下一唯一动作是
`PLAN_BOUNDED_ARCHITECTURAL_DECISION_DIRECT_COMPLEX_JOINT_VS_LIGHTGEN_INPUT_ALIGNMENT`；在该规划决策前，
不得自动开始另一个 phase family、训练或 generation。

## N3 minimal q-conditioned phase-coupling contract definition（2026-09-10）

按上一 Gate 的唯一下一动作完成 definition-only consumer。合同 artifact 为
[`n3_minimal_q_conditioned_phase_coupling_contract.v1.json`](../outputs/20260910_v2/runs/n3_minimal_q_conditioned_phase_coupling_contract_v1_attempt_01/n3_minimal_q_conditioned_phase_coupling_contract.v1.json)，
SHA-256=`14f4f3cd5d1046eb648b3410593d778910f8b8d33262528f9e0f12ba94250434`，
status/result=`valid_completed/READY`，classification=
`DEFINE_ONE_MINIMAL_Q_CONDITIONED_PHASE_COUPLING_MODEL_CONTRACT_V1_READY`。

合同只冻结一个项目模型假设：`Q_CONDITIONED_NEAREST_PHASE_DONOR_V1`。每折先按原有
`EMPIRICAL_GAUSSIAN_COPULA` 语义从 4000 个 fitting q rows 生成 `q_new`，再在同折 fitting-only
donor table 中，以 fit-derived 100-D Gaussianized q representation 的普通 Euclidean distance
选择恰好一个最近 donor（`K=1`）；完全相同距离按 stable fitting-row identity 的 UTF-8 字节序
选择最小者。donor 复制完整 100 维 frozen relative-phase vector，构造
`z_new=q_new*exp(i*phi_donor)`，不做逐坐标混合、相位插值或角度平均。

合同绑定既有 `PROJECT_COMPLEX_OLS_V1`、V2 development/five-fold authority、冻结 Gaussian
producer、上一轮完整复 OLS joint contract 与其 `FULL_COMPLEX_ROW_EMPIRICAL_BOOTSTRAP_CONTROL=5/5`
结果；若 q-distance 的唯一经验 CDF cross-section、fold isolation、donor identity 或旧 control
语义无法保持，则 `HARD STOP`。未来 candidate evaluator 仍固定为 canonical complex → float64
`[Re,Im]`、唯一 HARD primary=`energy_distance`、`B=999`、per-fold alpha=`0.01`、strict `5/5`；
secondary diagnostics 仅 `DESCRIPTIVE_ONLY`。

本次没有加载 q/complex/sealed values，没有执行 heldout evaluation、CUDA 或 training；machine
governance SHA 保持不变，`sealed consumed=true` 且不可复用，selected producer 与
`training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready` 均保持。
该合同是项目 model assumption，不是 Chen/LightGen phase-generation mechanism；当时唯一下一动作是
`EXECUTE_N3_Q_CONDITIONED_NEAREST_PHASE_DONOR_V1`，随后已由下一节记录的 execution consumer 消费。

## N3 q-conditioned nearest phase-donor execution（2026-09-10）

按 READY contract 的唯一授权动作完成一次且仅一次 candidate execution。冻结合同 SHA 为
`14f4f3cd5d1046eb648b3410593d778910f8b8d33262528f9e0f12ba94250434`；report 为
[`n3_q_conditioned_nearest_phase_donor_report.v1.json`](../outputs/20260910_v2/runs/n3_q_conditioned_nearest_phase_donor_v1_attempt_01/n3_q_conditioned_nearest_phase_donor_report.v1.json)，
SHA=`d17a472558349b043da07bc51310e7a350263608c059f37361237b1000318bcb`；lineage 为
[`n3_q_conditioned_nearest_phase_donor_lineage.v1.json`](../outputs/20260910_v2/runs/n3_q_conditioned_nearest_phase_donor_v1_attempt_01/n3_q_conditioned_nearest_phase_donor_lineage.v1.json)，
SHA=`735fc652d334b2c4cc2a7b70a57c100d242149a17f72bb9d82a2de5b92f79c67`。

结果为 `VALID_COMPLETED`，分类为
`N3_COMPLEX_OLS_Q_CONDITIONED_NEAREST_PHASE_DONOR_NOT_SUPPORTED`。上一轮
`FULL_COMPLEX_ROW_EMPIRICAL_BOOTSTRAP_CONTROL` 的 `5/5` authority 原样复用且未重跑；唯一
candidate `Q_CONDITIONED_NEAREST_PHASE_DONOR_V1` 为 `0/5`。五折均为 calibration-valid，且
`T_candidate` / `exceedance` / `p_MC` 如下：

| fold | T_candidate | exceedance | p_MC | pass |
| ---: | ---: | ---: | ---: | :---: |
| 0 | 0.1324989195 | 0 | 0.001 | false |
| 1 | 0.1422853270 | 0 | 0.001 | false |
| 2 | 0.1595676023 | 0 | 0.001 | false |
| 3 | 0.1545405207 | 0 | 0.001 | false |
| 4 | 0.1536020389 | 0 | 0.001 | false |

每折生成总量为 `999800`，unique donor 为 `3940–3957`，nearest Gaussianized-q distance
中位数约 `8.706–8.712`；这些 donor 统计和 q-phase/complex secondary diagnostics 均为
`DESCRIPTIVE_ONLY`，不参与 pass/fail。该结果只说明当前 K=1、fit-derived Gaussianized-q
nearest、完整 relative-phase donor 继承不足以通过当前 joint Gate；不说明所有 conditional
phase models、振幅 producer、论文机制或物理 phase-SLM 均失败。

本次不重用 sealed、不改 selected producer、不执行 K/metric search，不进入 generator/image/
optical forward、CUDA、training 或 formal promotion。machine governance 未修改；该时点下一规划边界为
`DEFINE_ONE_HIGHER_EXPRESSIVITY_Q_CONDITIONED_PHASE_MODEL_CONTRACT_V1`，随后由下节 definition-only consumer
完成；candidate execution 不自动开始。

## N3 higher-expressivity q-conditioned complex-phasor contract definition（2026-09-10）

按上一 Gate 的唯一规划动作完成 definition-only consumer。合同 artifact 为
[`n3_q_conditioned_complex_phasor_linear_residual_bootstrap_contract.v1.json`](../outputs/20260910_v2/runs/n3_q_conditioned_complex_phasor_linear_residual_bootstrap_contract_v1_attempt_01/n3_q_conditioned_complex_phasor_linear_residual_bootstrap_contract.v1.json)，
SHA-256=`8f19c3dbb6b9bc17560276e7f1cf61004fd4b0583a3c8a15792b4a86fa05c29f`，
status/result=`valid_completed/READY`，classification=
`DEFINE_ONE_HIGHER_EXPRESSIVITY_Q_CONDITIONED_PHASE_MODEL_CONTRACT_V1_READY`。

合同只冻结一个候选：`Q_CONDITIONED_COMPLEX_PHASOR_LINEAR_RESIDUAL_BOOTSTRAP_V1`。输入为同折
fitting-only 的 `FIT_EMPIRICAL_GAUSSIANIZED_Q_EUCLIDEAN_V1` 坐标 `x`；目标为完整 100-D
complex unit phasor `U=exp(i*phi)`，不回归 wrapped angles。固定设计为 `[1,x]` 的 101 列
multivariate OLS，CPU float64 thin-SVD pseudoinverse，`rcond=1e-12`；rank 低于 101 即
`HARD_STOP`，不加 ridge、正则、rank selection 或 model-order search。

残差池是完整 100-D complex residual vector，按行 uniform with replacement 抽取一个；生成的
`V=mu+R_d` 逐坐标做 `V/abs(V)`，任何非 finite 或 `abs(V)<=1e-12` 都是 `INVALID_HARNESS`，不重试或修复。
随后保持 `A(q)=q` 与既有 `PROJECT_COMPLEX_OLS_V1` gauge。未来 evaluator 仍固定 `B=999`、
per-fold alpha=`0.01`、`energy_distance` 为唯一 HARD primary、严格 `5/5`；control=`5/5`
只语义等价复用，不重跑。该候选不是 Chen/LightGen phase mechanism 声明。

本次未加载 q/complex/sealed values，未执行 heldout evaluation、CUDA、training 或 generation；machine
governance 未修改，`training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready`
保持。唯一下一动作是
`EXECUTE_N3_Q_CONDITIONED_COMPLEX_PHASOR_LINEAR_RESIDUAL_BOOTSTRAP_V1`，不自动开始。

## 阅读顺序

1. 本页：主线、当前成果、缺口与执行位置。
2. [当前交接](status/current_handoff.md)：本次目标变化及下一任务需要解决的具体差异。
3. [项目记忆](PROJECT_MEMORY.md)：长期目标与边界。
4. [正式 v2 架构](architecture/v2_runtime.md)：兼容 runtime；[实验历史](status/experiment_history.md)：逐次证据溯源。

以下保留整理前正文，避免丢失已有未提交记录和历史链接。其中“最新”“唯一下一步”、
未消费 sealed、未选 producer 等表述只属于各自记录时点；当前状态以本页上方及对应机器证据为准。
以后新结果更新上方摘要，日期化细节进入实验历史，不继续堆叠到首页。

<details>
<summary>展开整理前状态记录（含历史结果、合同与已被取代的下一步）</summary>

更新时间：2026-09-10

Git 状态：`research/lightgen-ols-native-semantic-generation-v1`；精确 HEAD、dirty scope
和 clean/dirty 状态以本地 Git 与 bridge manifest 为准；本轮 N3 attribution-contract definition、
attribution consumer、mixed component-swap diagnostic contract、empirical-fit ceiling diagnostic、
development-sampling stationarity diagnostic、source-position-balanced protocol/empirical-ceiling machine artifacts、
    N4 q-producer comparison、fixed-R Student-t producer comparison、q-producer self-null/guardrail calibration、V3 frozen-candidate replay、selected-producer full-development fit/state freeze、sealed-confirmation execution 与状态文档变更后尚未执行 bridge refresh，不声明 publication
为 `current_clean`。

## 当前决策摘要

正式 reconstruction mainline 已在当前 evidence boundary 收口；独立的
OLS-native semantic-generation research line 仍在活动分支
`research/lightgen-ols-native-semantic-generation-v1` 上，但不构成 formal v3。
N1/N2 已完成，N3 producer baseline、selector calibration 与两候选 minimal
producer comparison 均已完成但未支持新 producer family；N3 copula failure attribution
已在 V2 completeness contract 下完成 development-only read-only measurement，结果为
`N3_COPULA_FAILURE_MARGINAL_AND_DEPENDENCE_MIXED`；随后 oracle-guided minimal intervention
comparison 的 V1 在 Candidate M 的冻结非负输出边界以 `INVALID_HARNESS` blocked，未产生科学
分类或 selected candidate；随后 V2 已冻结零截断 M2 支持修复 contract，并完成一次新的
M2-vs-D comparison，结果为 `NO_MINIMAL_INTERVENTION_SUPPORTED`（M2=`0/5`、D=`1/5`，无
  selected candidate）；随后 N4 已按冻结的 V2 authority 完成唯一一次 q-producer
  development comparison，三候选均未达到严格 `5/5` fold support（copula=`4/5`，fold 4
  失败）；随后按独立 Gate 执行 fixed-R Student-t copula producer comparison，五折中 `4/5`
  兼容（fold 2 的 `energy_distance`、`marginal_mean_l2`、`marginal_energy_distance_mean`
  失败），同样未达到严格 `5/5`；随后 self-null calibration 完成固定 `200` trials/cell，
  aggregate 为 `N3_Q_COMPATIBILITY_SELF_NULL_MIXED_OR_UNRESOLVED`；再后 guardrail calibration
  已按独立合同完成，得到 `N3_Q_COMPATIBILITY_GUARDRAIL_CALIBRATION_READY`，并只保留
  `energy_distance` 为 HARD primary，其他七项转为 `DESCRIPTIVE_ONLY`。随后仅消费既有 Gaussian
  与 fixed-R Student-t 的 exact frozen candidate banks，完成 V3 evaluator replay：Gaussian 在
  primary `energy_distance` 下为 `5/5` development-supported；Student-t 为 `4/5`，fold 2
  primary 失败，因此不支持。该 replay 只改变 evaluator authority，不改写历史 V2 分类；随后已完成
   producer-selection/freeze contract definition，唯一候选为 Gaussian；随后已按该契约完成一次 all-5000
   development fit 并发布 immutable producer state。当前 development authority 的
   `selected_producer=EMPIRICAL_GAUSSIAN_COPULA`、`frozen_producer_state` 已绑定，但这不等于
   training、formal promotion 或 generation readiness。N5 与 generation 仍 blocked，N6 保持独立冻结；
   随后已完成 sealed confirmation contract definition，并按该 contract 完成一次且仅一次
   sealed confirmation execution，结果为 `N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_PASSED`；随后按下一
   规划边界完成了 definition-only 的 complete-complex-OLS amplitude/phase joint-compatibility
   contract，结果为 `N3_COMPLEX_OLS_AMPLITUDE_PHASE_JOINT_COMPATIBILITY_CONTRACT_V1_READY`；
   下一唯一允许动作是 `EXECUTE_N3_COMPLEX_OLS_AMPLITUDE_PHASE_JOINT_COMPATIBILITY_V1`，不自动执行。

## N3 selected producer sealed confirmation execution（2026-09-10）

按唯一授权动作 `EXECUTE_N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_V1`，在冻结 contract、
producer state 与 reservation authority 不变的前提下完成一次正式 CPU-only execution。lineage 为
[`n3_selected_producer_sealed_confirmation_lineage.v1.json`](../outputs/20260910_v2/runs/n3_selected_producer_sealed_confirmation_v1_attempt_01/n3_selected_producer_sealed_confirmation_lineage.v1.json)，
SHA-256=`a8e8dc3ed3b4c2a5eb274aa0d089e6697aed8fdda83bf513c36bfdf548210bb7`，
status/result=`valid_completed/VALID_COMPLETED`。

在 sealed exposure 前完成 `B=999` model-self calibration 与固定的 `4×200` observed MODEL
banks，并记录 `SEALED_CONFIRMATION_ARMED`；随后按 write-once 规则将 marker 置为
`consumed=true`，精确 materialize `500` 行（`50/class`）。primary V3 `energy_distance`
结果为 `T_sealed=0.04642494058877489`，Monte-Carlo exceedance=`473`，
`p_MC=0.474`（`alpha=0.05`），因此通过。marker SHA=`48faa3bc6aadc1d5210b164ed15d09ff465cd8683726cac1d1a86d7fe7b8dbe6`；
calibration、MODEL banks、sealed q 与 q-manifest 均已冻结并写入同一 run 目录。

本结果只支持该冻结 Gaussian q/amplitude producer 在这一个独立 sealed contract 下通过；
不证明 complete complex OLS、amplitude-phase joint compatibility、semantic/optical generation
或 population/OOD generalization。`selected_producer=EMPIRICAL_GAUSSIAN_COPULA`、
`n3_amplitude_producer_confirmation=PASSED`，但 `training_authorized=false`、
`formal_promotion=false`、`generation_readiness=not_ready`。N3 amplitude/q producer work
到此关闭；下一步仅进入 complete-complex-OLS contract planning，不自动执行。

## N3 complete complex OLS amplitude/phase joint-compatibility contract definition（2026-09-10）

按当前唯一规划边界完成 definition-only contract consumer。机器契约为
[`n3_complex_ols_amplitude_phase_joint_compatibility_contract.v1.json`](../outputs/20260910_v2/runs/n3_complex_ols_amplitude_phase_joint_compatibility_contract_v1_attempt_01/n3_complex_ols_amplitude_phase_joint_compatibility_contract.v1.json)，
SHA-256=`52e07263343d765db13022fbcf56f74e7431a9dced16f8bfa31ca07ecc6bd726`，
status/result=`valid_completed/READY`，classification=
`N3_COMPLEX_OLS_AMPLITUDE_PHASE_JOINT_COMPATIBILITY_CONTRACT_V1_READY`。

契约绑定 `PROJECT_COMPLEX_OLS_V1` 的既有 `R3_COMPLEX` 语义、`N3_DEVELOPMENT_BANK_V2`
source/q/five-fold authority（5000 rows、每折 4000 fit/1000 heldout、assignment SHA=
`43ee0ce28e143fffdf2bcac98394625ccd766166ea717f23db66ed8770ef18a9`）以及
`N3_SELECTED_EMPIRICAL_GAUSSIAN_COPULA_PRODUCER_V1` state SHA=
`752db16087e7bdf8343e4499202182370b376a1a2d5925a383f411fd4a15bb88`。V2 q→complex-amplitude
映射固定为 `A_j(q)=q_j=abs(c_j/sqrt(s_j))`，复数输入为 `torch.complex64`、规范化后为
`torch.complex128`，维度 `d=100`。

相位语义 authority 为 `OBSERVED_RELATIVE_PHASE_INHERITANCE_V1`，在本契约中冻结为既有
`GLOBAL_PHASE_GAUGE_CONFIRMED` 商空间的确定性数值 cross-section：每行取最大幅值的最小索引
作为 anchor，保留完整 100 维相对相位，使用 `atan2` 的 `[-pi,pi]` 区间；exact-zero
coordinate 为 `INVALID_HARNESS`，正的 near-zero 不阈值化、不 mask、不 epsilon 替换。该
cross-section 不是新的物理 phase reference。

评估顺序固定为先执行不可选的
`FULL_COMPLEX_ROW_EMPIRICAL_BOOTSTRAP_CONTROL`（每折 fitting complex OLS 完整行有放回抽样，
保持 amplitude/phase 配对），control 只有 `5/5` 才能解释 candidate。唯一 candidate 为
`OBSERVED_PHASE_INDEPENDENT_INHERITANCE_V1`：每折只用 fitting q/complex rows，独立有放回抽取
一个完整 phase vector 后与 Gaussian q amplitude 合成，再按既有 global-phase quotient canonicalize；
不使用 heldout phase donor，不做 q-dependent phase selection。

primary 仅为 gauge-canonical `[Re(z),Im(z)]` `torch.float64` embedding 上的 `energy_distance`；
amplitude/q、phase 和 amplitude-phase dependence 仅 `DESCRIPTIVE_ONLY`。每折/每类先完成同结构
model-self calibration，`4×200` model banks、`B=999`、Monte-Carlo ties 纳入计数，family alpha
`0.05` 经 Bonferroni 固定为 per-fold alpha `0.01`，严格 `5/5`。本轮没有读取 q/complex 值、没有
执行科学 joint evaluation、没有复用已消耗 sealed；旧 OLS artifact 只有历史 1000 行，V2 对应的
5000 行 complex OLS 只冻结未来确定性 source-row materialization，未在定义阶段虚构 artifact。

定义完成时仍为 `training_authorized=false`、`formal_promotion=false`、
`generation_readiness=not_ready`；sealed 维持已消费且不可 reset/retry。该 READY 只允许随后执行
一次冻结的 joint-compatibility consumer，不代表 joint-complex support 或任何
semantic/physical generation readiness。

## N3 complete complex OLS amplitude/phase joint-compatibility execution（2026-09-10）

按唯一授权动作
`EXECUTE_N3_COMPLEX_OLS_AMPLITUDE_PHASE_JOINT_COMPATIBILITY_V1` 完成一次且仅一次
CPU-only formal diagnostic execution。执行 report 为
[`n3_complex_ols_amplitude_phase_joint_compatibility_report.v1.json`](../outputs/20260910_v2/runs/n3_complex_ols_amplitude_phase_joint_compatibility_v1_attempt_01/n3_complex_ols_amplitude_phase_joint_compatibility_report.v1.json)，
SHA-256=`9989b89dea8a8d299da9a89b8c5566376a27a21c250a5ca22c07202dfa938429`；lineage 为
[`n3_complex_ols_amplitude_phase_joint_compatibility_lineage.v1.json`](../outputs/20260910_v2/runs/n3_complex_ols_amplitude_phase_joint_compatibility_v1_attempt_01/n3_complex_ols_amplitude_phase_joint_compatibility_lineage.v1.json)，
SHA-256=`238d6638a16fcfa86e48216db2b886b45c9038f8ea017b7f51a55c1f18090aa0`。

消费者先执行不可选的 `FULL_COMPLEX_ROW_EMPIRICAL_BOOTSTRAP_CONTROL`，五折均通过：
`5/5`，每折 `B=999`、primary=`energy_distance`、per-fold alpha=`0.01`。随后执行唯一
candidate `OBSERVED_PHASE_INDEPENDENT_INHERITANCE_V1`；五折均失败，`0/5`，因此分类为
`N3_COMPLEX_OLS_INDEPENDENT_PHASE_INHERITANCE_NOT_SUPPORTED`。本结果拒绝的是当前
“Gaussian q/amplitude + 独立继承完整 observed phase vector”的联合兼容性假设；不证明
不存在任何 q-conditioned phase model，也不证明论文方法或物理系统失败。

V2 exact complex materialization 为 `torch.complex64`、`[5000,100]`，并满足
`q=abs(z)` 位级一致；旧历史 1000-row OLS artifact 未作为本次 row source。sealed
confirmation 维持 `consumed=true`、`reused=false`，未载入 sealed values；
`selected_producer=EMPIRICAL_GAUSSIAN_COPULA` 不变，`training_authorized=false`、
`formal_promotion=false`、`generation_readiness=not_ready` 均保持。唯一下一动作是
`DEFINE_ONE_MINIMAL_Q_CONDITIONED_PHASE_COUPLING_MODEL_CONTRACT`，不自动定义或执行。

随后只对既有五折 development protocol 执行了一次 complete-row empirical-fit joint bootstrap
ceiling diagnostic：每折从 exact `800` 个 fitting q rows 中有放回抽取完整行，未拟合新的
producer family。该经验联合分布 ceiling 仅在 `3/5` folds 通过冻结的 8 个 compatibility
guardrails，因此分类为 `N3_FIT_HELDOUT_EMPIRICAL_CEILING_NOT_SUPPORTED`。这是 FACT；其
失败与 fit-to-heldout distribution stability / finite-development sampling limitation 一致，
但不是“没有 population-level producer”的证明，也不授权升级 producer family。随后
development-sampling stationarity diagnostic 判定 `N3_DEVELOPMENT_SAMPLING_STABILITY_LIMITATION_SUPPORTED`；
再后冻结 deterministic class/source-position-balanced development protocol 并完成唯一一次新的
empirical ceiling validation，结果为 `SOURCE_POSITION_BALANCED_DEVELOPMENT_PROTOCOL_NOT_SUPPORTED`
（`0/5`）。在该结果之后，按单一 bounded authorization 冻结并物化 replacement
`N3_DEVELOPMENT_BANK_V2`，再完成唯一一次 V2 empirical ceiling validation；V2 结果为
`N3_DEVELOPMENT_BANK_V2_VALIDATED`（`5/5`），因此 5000-row bank 可作为后续 N3 producer
  development selection 的 development authority。该结果仍不等于 producer support、producer
  selection、training 或 generation authorization；随后唯一一次 N4 producer-development
  comparison 已执行并关闭。

最新 N3 selector calibration V2 classification=`N3_ATTRIBUTION_SELECTOR_CALIBRATION_MIXED_V2`。
冻结 selector 在独立 synthetic confirmation 上 NULL=`0/200` false positives，
partition=`100/100`，radial=`100/100`，nonlinear=`82/100`，local=`78/100`；local
未达到预注册 `0.80` correct-rate target，故不能宣布 calibrated。selector 不能重新解释真实
N3 failure，也不能授权新模型族、generation 或 formal promotion；V2 attribution consumer
已在冻结契约下完成一次 development-only measurement。N4 由后续独立 V2 authority comparison
完成且未选出 producer。当前 flags
仍为 `training_authorized=false`、`generation_readiness=not_ready`、
`formal_promotion=false`；本 selector refinement line 在 MIXED 边界关闭，不创建 V3。
对应 machine evidence 为
[`n3_attribution_selector_v2_calibration_report.v1.json`](../outputs/20260908_v2/runs/lightgen_n3_attribution_selector_v2_calibration_attempt_01/n3_attribution_selector_v2_calibration_report.v1.json)，
frozen spec SHA=`9ee5b7ad64221b323f1b182c2f4d74edddfad89599bdf30b352ef33cc5b4c25b`，
report SHA=`77c0a9f632b6870579a338d84dd6650530849fa6b74892acb1345567ffa8dae5`。

## N3 q-compatibility guardrail calibration V1（2026-09-09）

在 self-null V1 的 MIXED 边界之后，按独立 Gate
`CALIBRATE_N3_Q_COMPATIBILITY_GUARDRAIL_GATE_V1` 完成唯一一次 calibration。预模拟合同为
[`n3_q_compatibility_guardrail_gate_contract.v1.json`](../outputs/20260909_v2/runs/n3_q_compatibility_guardrail_gate_contract_v1_attempt_01/n3_q_compatibility_guardrail_gate_contract.v1.json)，
SHA-256=`a87979f9689159eee6d769f10eb6b7e30bcdc9803c4b72b108225a6ea22935b4`。合同冻结两个既有
producer、五折、每 producer/fold `200` 次（总计 `2000`）、`1000/200` 与 `200/200` 样本结构、
现有 V2 raw-q evaluator、单侧 `95%` exact Clopper–Pearson 上界以及最坏 producer×fold
上界 `≤0.01` 的 HARD eligibility rule；五折可靠性按配对 trial index 直接计算，目标为两类
producer 均 `≥0.80`。未重跑历史 heldout，未读取 sealed。

唯一 calibration report 为
[`n3_q_compatibility_guardrail_calibration_report.v1.json`](../outputs/20260909_v2/runs/n3_q_compatibility_guardrail_gate_v1_attempt_01/n3_q_compatibility_guardrail_calibration_report.v1.json)，
SHA-256=`7135b07fd532ee5dd2dba7900ad6028c9a6d558924691e4d65eac56330367789`，status/result=`valid_completed/VALID_COMPLETED`，
classification=`N3_Q_COMPATIBILITY_GUARDRAIL_CALIBRATION_READY`。各 metric 的 pooled false-fail
与最坏 cell 上界如下：

| metric | pooled false-fail | worst-cell CP upper bound | V3 role |
|---|---:|---:|---|
| `energy_distance` | `0/2000 = 0.0000` | `0.014867`（primary excluded from selection） | `HARD_PRIMARY_UNCHANGED` |
| `covariance_relative_frobenius` | `0/2000 = 0.0000` | `0.014867` | `DESCRIPTIVE_ONLY` |
| `norm_energy_distance` | `101/2000 = 0.0505` | `0.101357` | `DESCRIPTIVE_ONLY` |
| `squared_norm_energy_distance` | `101/2000 = 0.0505` | `0.101357` | `DESCRIPTIVE_ONLY` |
| `marginal_mean_l2` | `7/2000 = 0.0035` | `0.038310` | `DESCRIPTIVE_ONLY` |
| `marginal_std_l2` | `2/2000 = 0.0010` | `0.023498` | `DESCRIPTIVE_ONLY` |
| `marginal_energy_distance_mean` | `0/2000 = 0.0000` | `0.014867` | `DESCRIPTIVE_ONLY` |
| `marginal_energy_distance_max` | `14/2000 = 0.0070` | `0.045180` | `DESCRIPTIVE_ONLY` |

当前 V2 完整八项 gate 的直接配对五折通过率为 Gaussian `144/200=0.72`、Student-t
`141/200=0.705`；按冻结规则生成的 V3 primary-only gate 两者均为 `200/200=1.0`，达到
`0.80` 目标。因此 V3 evaluator authority 为
[`n3_q_compatibility_evaluator_v3.v1.json`](../outputs/20260909_v2/runs/n3_q_compatibility_guardrail_gate_v1_attempt_01/n3_q_compatibility_evaluator_v3.v1.json)，
SHA-256=`4fbc831e383897802134ea2ce1aaef6446629130a7c799630254a86b8dcff3cc`。V3 保持 primary
`energy_distance`、metric formulas、sample-size semantics、development/heldout authorities
与 finite-sample limitations 不变；它不是 producer support、selection、training 或 generation
授权。N4、Gaussian 与 Student-t 历史分类不变；在 replay 前的 hard blocker 是
`N4_Q_PRODUCER_COMPARISON_NO_CANDIDATE_SUPPORTED`，不会因为 V3 READY 自动升级 producer；
随后 replay 与 producer-selection/freeze contract 的当前结果和下一动作见下两节。
`selected_producer=null`、
`training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready`，sealed
仍为 `RESERVED_NOT_MATERIALIZED/consumed=false`。
治理入口当前为 `active_objective=N3_Q_PRODUCER_V3_DEVELOPMENT_SUPPORTED`、
`hard_blocker=N3_Q_PRODUCER_SELECTION_FREEZE_REQUIRED`；这只是 selection/freeze 前置边界，
不构成 downstream authorization。

## N3 frozen candidate V3 evaluator replay（2026-09-09）

按独立 Gate `APPLY_N3_Q_COMPATIBILITY_EVALUATOR_V3_TO_FROZEN_CANDIDATES_V1`，在已验证的
`N3_DEVELOPMENT_BANK_V2` source/q/five-fold/ceiling authority 上，只重放既有 N4 Gaussian
与 fixed-R Student-t 的 exact fit/sample artifacts。V3 evaluator、primary `energy_distance`
实现、q95/reference construction、sample sizes、folds 与 DATA/MODEL rows 均未改变；没有
refit、resampling、seed retry、heldout tuning 或 sealed access。

唯一 replay report 为
[`n3_q_compatibility_evaluator_v3_frozen_candidate_replay_report.v1.json`](../outputs/20260909_v2/runs/n3_q_compatibility_evaluator_v3_frozen_candidate_replay_v1_attempt_01/n3_q_compatibility_evaluator_v3_frozen_candidate_replay_report.v1.json)，
SHA-256=`cf6e6d62e6d2a7b35822d2d3727d08e5cbd59d51778bf794ec30ce297ffd0195`，status/result=`valid_completed/VALID_COMPLETED`，
classification=`N3_Q_COMPATIBILITY_EVALUATOR_V3_FROZEN_CANDIDATE_REPLAY_COMPLETE`。对应 lineage
manifest 为
[`n3_q_compatibility_evaluator_v3_frozen_candidate_replay_lineage.v1.json`](../outputs/20260909_v2/runs/n3_q_compatibility_evaluator_v3_frozen_candidate_replay_v1_attempt_01/n3_q_compatibility_evaluator_v3_frozen_candidate_replay_lineage.v1.json)，
SHA-256=`d49800c12ce5f7e4931aca301410d5f0c47ca301350324fdfae3a73526d19ee4`；lineage 绑定
20 个历史 fit/sample artifacts，且每个文件 SHA 可复核。

| candidate | V3 primary fold pattern | V3 classification | historical V2 classification |
|---|---|---|---|
| empirical Gaussian copula | `[true,true,true,true,true]` (`5/5`) | `N3_EMPIRICAL_GAUSSIAN_COPULA_V3_DEVELOPMENT_SUPPORTED` | N4 `4/5`, `NOT_SUPPORTED` |
| fixed-R Student-t copula | `[true,true,false,true,true]` (`4/5`) | `N3_FIXED_R_STUDENT_T_COPULA_V3_NOT_SUPPORTED` | V2 `4/5`, `NOT_SUPPORTED` |

Gaussian 五折 primary `energy_distance` / q95 为 `(0.0492015423,0.0751950320)`、
`(0.0531102928,0.0726604040)`、`(0.0463874651,0.0667172126)`、
`(0.0399628736,0.1176919572)`、`(0.0454373059,0.0759391666)`；fold 4 的
`norm_energy_distance` 与 `squared_norm_energy_distance` 只作为 descriptive failure，不能
否决 V3 support。Student-t fold 2 primary 为 `0.0708070262` 对 q95 `0.0667172126`，故严格
`5/5` 不支持；其余七项全部保留为 `DESCRIPTIVE_ONLY`，其历史阈值 pass/fail 只作报告。

历史 V2 的 Gaussian=`N4_Q_PRODUCER_COMPARISON_NO_CANDIDATE_SUPPORTED`、Student-t=
`N3_FIXED_R_STUDENT_T_COPULA_NOT_SUPPORTED` 均保持 immutable。该结果仅支持“冻结候选样本
在 V2 development bank 上通过校准 V3 raw-q compatibility”的有限 claim，不支持 stationarity、
OOD、sealed confirmation、complex OLS、phase/semantic/image/optical generation、training 或
promotion。`development_supported_candidate` 仅为 Gaussian，未执行 post-hoc selection；
`selected_producer=null`、`training_authorized=false`、`formal_promotion=false`、
`generation_readiness=not_ready`，sealed 仍为 `RESERVED_NOT_MATERIALIZED/consumed=false`。
producer-selection/freeze contract 已在下一节定义，但不自动 final fit，不消费 sealed。

## N3 selected producer freeze contract（2026-09-09）

按独立 Gate `DEFINE_N3_SELECTED_PRODUCER_FREEZE_CONTRACT_V1`，在不读取 q 数值、不拟合、不采样、
不访问 sealed 的前提下，完成唯一的 selected-producer definition。机器契约为
[`n3_selected_producer_freeze_contract.v1.json`](../outputs/20260909_v2/runs/n3_selected_producer_freeze_contract_v1_attempt_01/n3_selected_producer_freeze_contract.v1.json)，
SHA-256=`4d8d3f5dc81a7eff2d92a0acde0a0d0639788bb0dee75e6d88db745f642b60a3`，
status/result=`valid_completed/VALID_COMPLETED`，classification=
`DEFINE_N3_SELECTED_PRODUCER_FREEZE_CONTRACT_V1_READY`。

selection 只来自冻结 V3 primary `energy_distance` 的支持结果：
`EMPIRICAL_GAUSSIAN_COPULA` 为唯一 `5/5` development-supported candidate；fixed-R
Student-t 为 `4/5`，保持 `NOT_SUPPORTED`。未使用 descriptive V3 guardrails 做 tie-break，
也未改写历史 V2：Gaussian 仍为 `N4_Q_PRODUCER_COMPARISON_NO_CANDIDATE_SUPPORTED` 的
`4/5`，Student-t 仍为 `N3_FIXED_R_STUDENT_T_COPULA_NOT_SUPPORTED` 的 `4/5`。

契约把后续唯一拟合动作固定为 `N3_DEVELOPMENT_BANK_V2` 全部 `5000` 行、`500/class`、
`R2_AMPLITUDE`、CPU `float64`、不使用 heldout/fold 子集、不以 labels 条件化 producer；
五折证据只承担先前 selection 角色。经验边际、stable average-rank、
`(rank-0.5)/n_fit`、`ndtri`、`R_emp`、解析 shrinkage、Cholesky/正特征值和 unit-diagonal
fail-closed 规则均按既有 Gaussian candidate 冻结，没有新增 model/hyperparameter choice。

未来 state 的语义 authority 为 `N3_SELECTED_EMPIRICAL_GAUSSIAN_COPULA_PRODUCER_V1`，
schema 为 `n3_selected_empirical_gaussian_copula_producer_state.v1`；契约同时冻结了
`torch.save` mapping、CPU `float64` tensors、一次性 SHA-256、以及只依赖 frozen state、
requested sample count 和显式 RNG namespace/seed 的独立 sampling interface。但本轮
`frozen_producer_state=null`，未产生 all-5000 fit/state/sample artifact。当前仍为
`training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready`，
sealed 为 `RESERVED_NOT_MATERIALIZED/consumed=false`。下一步且仅下一步为
`EXECUTE_N3_SELECTED_PRODUCER_FULL_DEVELOPMENT_FIT_V1`；该动作完成前不得打开 sealed confirmation。

## N3 selected producer full development fit（2026-09-10）

按 `DEFINE_N3_SELECTED_PRODUCER_FREEZE_CONTRACT_V1` 的唯一后续动作
`EXECUTE_N3_SELECTED_PRODUCER_FULL_DEVELOPMENT_FIT_V1`，完成了一次且仅一次
`N3_DEVELOPMENT_BANK_V2` 全量 `5000` 行、`500/class`、`R2_AMPLITUDE`、CPU `float64`
拟合。预检先核对了 freeze contract、V2 source/q authority、唯一/平衡行身份、既有
manifest 的零 overlap 与 sealed reservation 元数据；q 只在预检通过后加载，未读取 sealed
q/image/label 值，未使用 fold/heldout 子集，也未以 labels 条件化 producer。

唯一 producer state 为
[`n3_selected_empirical_gaussian_copula_producer_state.v1.pt`](../outputs/20260910_v2/runs/n3_selected_producer_full_development_fit_v1_attempt_01/n3_selected_empirical_gaussian_copula_producer_state.v1.pt)，
semantic authority=`N3_SELECTED_EMPIRICAL_GAUSSIAN_COPULA_PRODUCER_V1`，SHA-256=
`752db16087e7bdf8343e4499202182370b376a1a2d5925a383f411fd4a15bb88`。对应 lineage/manifest 为
[`n3_selected_producer_full_development_fit_lineage.v1.json`](../outputs/20260910_v2/runs/n3_selected_producer_full_development_fit_v1_attempt_01/n3_selected_producer_full_development_fit_lineage.v1.json)，
SHA-256=`9244df01c95e44005207cb1dc188fc7de2961665e13055beebf55f0f5cdd2267`，
status/result=`valid_completed/VALID_COMPLETED`，classification=
`N3_SELECTED_PRODUCER_FULL_DEVELOPMENT_FIT_COMPLETED`。

拟合审计为：stable average-rank empirical marginal、`ndtri` Gaussianization、analytic
shrinkage；`R_shrunk` 最小特征值 `0.0401643351`，最大特征值 `18.4033362231`，
unit-diagonal error=`5.995204332975845e-15`（容差 `5.684341886080801e-14`）。state
落盘后 safe-load 与 tensor identity 均一致；standalone sampler 只依赖 state、显式
namespace/seed，并通过非科学性的 3-sample deterministic smoke。未执行 full-fit compatibility
评估、sealed confirmation、optical/semantic generation、CUDA、optimizer/backward、training
或 promotion。

因此当前仅将该 state 作为 development-only selected-producer authority：
`selected_producer=EMPIRICAL_GAUSSIAN_COPULA`、`training_authorized=false`、
`formal_promotion=false`、`generation_readiness=not_ready`，sealed 仍为
`RESERVED_NOT_MATERIALIZED/consumed=false`。唯一下一步是定义新的
`DEFINE_N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_CONTRACT_V1`；不得把本次 state freeze
解释为 confirmation、population generalization 或 optical readiness。

## N3 selected producer sealed confirmation contract（2026-09-10）

按唯一 Gate `DEFINE_N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_CONTRACT_V1`，在不读取、
materialize、decode、transform 或 evaluate sealed content 的前提下，完成了对冻结
`N3_SELECTED_EMPIRICAL_GAUSSIAN_COPULA_PRODUCER_V1` 的一次性独立 confirmation contract
定义。机器契约为
[`n3_selected_producer_sealed_confirmation_contract.v1.json`](../outputs/20260910_v2/runs/n3_selected_producer_sealed_confirmation_contract_v1_attempt_01/n3_selected_producer_sealed_confirmation_contract.v1.json)，
SHA-256=`e0e29a62bce4eaaa48346ce65d1d7d6dcc2892df4d8e9ace73681579232c1200`，
status/result=`valid_completed/READY`，classification=
`DEFINE_N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_CONTRACT_V1_READY`。

contract 精确绑定既有 sealed reservation
`N3_SEALED_CONFIRMATION_SOURCE_V1`（manifest SHA-256=
`6475474f157d31591a3b0252f27320323d84ea93803c61ee6a365420a8ebb08e`）、500 行、50/class、
固定 source-index order、identity/balance metadata 与 zero-overlap firewall；定义阶段只读
reservation manifest，不读 source images、OLS values 或 q values。source→q 语义沿用既有
V2 authority：`R2_AMPLITUDE=abs(c/sqrt(s))=abs(R3_COMPLEX)`、固定 OLS/calibration、CPU
`float64`、`[500,100]`、finite/nonnegative、无 clipping/samplewise normalization，未来
q artifact 必须带完整 row→q identity linkage。

primary confirmation statistic 只使用 V3 HARD `energy_distance`，沿用 V3 validated
implementation 与四个 `200`-row model-bank 的 `_mean_metrics` arithmetic mean；sealed DATA
为500行。开发期 DATA size 不同，因此不复用 V3 development q95，未来先以完全匹配
`500`-row DATA、4×200 model banks 的 model-self null 完成固定 `B=999` calibration，
再按 `p_MC=(1+count(T_b >= T_sealed))/(B+1)`、`alpha=0.05` 做一次 PASS/FAIL。其余七项
metric 仅为 `DESCRIPTIVE_ONLY`，不能 veto/rescue primary result。

contract 同时冻结了 disjoint RNG stream/seed rule、observed model-bank seeds、所有
pre-sealed arming conditions，以及 first protected-content read 前的 write-once
`consumed=true` transition。exposure 后任何失败均为
`SEALED_CONFIRMATION_INVALID_CONSUMED`，不得 reset 或 retry；exposure 前 failure 才能
保持 `consumed=false` 并 BLOCK。当前仍为 `RESERVED_NOT_MATERIALIZED/consumed=false`，
`training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready`。
唯一下一步且仅授权为 `EXECUTE_N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_V1`；不得自动
进入 semantic generation 或 training。

## N3 replacement development bank V2 与 empirical ceiling（2026-09-09）

按上述 bounded authorization，`N3_DEVELOPMENT_BANK_V2` 已完成 source-row definition、CPU q
materialization、五折 protocol freeze 与唯一一次 empirical ceiling validation。V2 source manifest
为 [`n3_development_bank_v2_source_manifest.v1.json`](../outputs/20260909_v2/runs/n3_development_bank_v2_source_manifest_v1_attempt_01/n3_development_bank_v2_source_manifest.v1.json)，
SHA-256=`4bc251af91b3a6cd0235fe113117443d871a564bd05634fa9e61b5ab0b36f812`，bank semantic
SHA=`bdb5a11c6800b4a5d0a994d5448bbf7de3e29c7bf8c0709f7d34be1cb31c91aa`，row manifest
SHA=`ba3149357dc90f14ad692a486593ce944d48453f2b561ad10a9323b9fe7fb957`。完整 eligible
`mnist_train` source domain 为 `48500` 行，逐类为 `[4782,5528,4818,4951,4709,4356,4801,5025,4692,4838]`；
排除 union 为 `11500` 行，old N3/S2/S1/sealed reservation overlap 均为 `0`。最终 bank 为
`5000` 行、`500/class`，selected source-index SHA=`8e6859cbe0ad10090b1d5d7c55de28232f9db30c45093078dbf29f8545473a0e`，
selected-label SHA=`d28f06d6ecde9d08797f954e60041f79b512e2ad61480e21d5702cb841fabef3`；选择只按
完整逐类 source-rank 的 deterministic evenly-spaced formula，未读取 q/OLS/image 值。

q authority 为 [`n3_development_bank_v2_q_materialization_manifest.v1.json`](../outputs/20260909_v2/runs/n3_development_bank_v2_q_materialization_v1_attempt_01/n3_development_bank_v2_q_materialization_manifest.v1.json)，
SHA-256=`422d8c65b78bc459197ed46504399cb9c87ba8f7184eb2405e0faa85353995d8`；q artifact
SHA=`fe48c8fdb9bb2bf732076fabdf475288baa6f7d628ca10039a83bbda54fcfcbf`，q tensor
SHA=`d660fd8dc449725edab081d00c8f553f151029cddef9210360a7a6fad4281209`。q 定义为 frozen
OLS authority 下的 `R2_AMPLITUDE = abs(c/sqrt(s)) = abs(R3_COMPLEX)`，CPU materialization 后
model state unchanged；未读取 sealed q/images。

五折 manifest 为 [`n3_development_bank_v2_fold_manifest.v1.json`](../outputs/20260909_v2/runs/n3_development_bank_v2_five_fold_protocol_v1_attempt_01/n3_development_bank_v2_fold_manifest.v1.json)，
SHA-256=`4898a0e477d583b9a9e60c83238e6d8524d43967efe24277a952089e8adc447e`，assignment
SHA=`43ee0ce28e143fffdf2bcac98394625ccd766166ea717f23db66ed8770ef18a9`。规则为 within-class
source-order rank mod 5；每折 exact `100/class` heldout、`1000` heldout、`400/class` fit、
`4000` fit，fit/heldout disjoint 且完整 bank 每折覆盖一次。

唯一 ceiling report 为 [`n3_development_bank_v2_empirical_ceiling_report.v1.json`](../outputs/20260909_v2/runs/n3_development_bank_v2_empirical_ceiling_v1_attempt_01/n3_development_bank_v2_empirical_ceiling_report.v1.json)，
SHA-256=`731cf826dc081e7cc5629790ba2fb74946f7cc41f1a7f0035bf3a3ca96a31826`，status/result=`VALID_COMPLETED`，
classification=`N3_DEVELOPMENT_BANK_V2_VALIDATED`，fold pattern=`[true,true,true,true,true]`。每折
使用 exact 1000-row heldout 对 200-row complete-q-row bootstrap bank；4 个 DATA-vs-MODEL、8 个
MODEL-vs-MODEL、pairs `(0,1),(2,3),(4,5),(6,7)`、frozen q95、raw-q ED 与全部 8 个 scalar
guardrails 均保持不变。由于新协议要求 unequal `1000/200` 行数，测量只移除了旧 pair-metric
的等行数计算前置条件，未改变 metric formula 或 decision rule；无 dimensionwise resampling、
seed retry 或 best-bank selection。

这是 development-only empirical-authority validation，不是 producer。未评估 learnable producer、
未选择 producer、未训练、未进入 decoder/evaluator、未消费 sealed confirmation，也未进入 N4/N5/N6
或 generation。old 1000-row authority remains historical and unchanged；sealed 仍为
`RESERVED_NOT_MATERIALIZED`/`consumed=false`，`selected_producer=null`、
`training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready` 保持不变。

## N4 q-producer development comparison（2026-09-09）

在冻结的 V2 source/q/five-fold/ceiling authority 上，先冻结候选 registry，再执行唯一一次
CPU-only、development-only 的 q-producer comparison。registry 为
[`n4_q_producer_candidate_registry.v1.json`](../outputs/20260909_v2/runs/n4_q_producer_candidate_registry_v1_attempt_02/n4_q_producer_candidate_registry.v1.json)，
SHA-256=`0329b23675a7d306c88f26c80a4757ce83d39c8c68b718bf7a1d966f53f4f27b`；它固定三个既有
q/amplitude 候选：shrinkage elliptical Gaussian、radial scale mixture、empirical marginal
Gaussian copula。候选 registry 在第一次 evaluator call 前冻结；旧 attempt-01 为 write-once
未消费 provenance，attempt-02 才是本次 comparison 使用的 registry。

唯一 comparison report 为
[`n4_q_producer_comparison_report.v1.json`](../outputs/20260909_v2/runs/n4_q_producer_development_comparison_v1_attempt_02/n4_q_producer_comparison_report.v1.json)，
SHA-256=`393d84617ec7b1e6c60b8c654b859dc3afe822f5dee207ca93816e232fea78be`，status/result=`VALID_COMPLETED`，
classification=`N4_Q_PRODUCER_COMPARISON_NO_CANDIDATE_SUPPORTED`。shrinkage elliptical Gaussian
与 radial scale mixture 均为 `0/5`；empirical marginal Gaussian copula 为 `4/5`，fold 4 的
`norm_energy_distance` 与 `squared_norm_energy_distance` 未通过。因此 copula 虽为最接近候选，
仍不能按严格 all-five-fold 规则获得支持或被选择；没有 best-of-N、heldout tuning、seed retry
或 post-result resampling。总 evaluator exposure 为 `15/15`，registry SHA 前后不变，adaptive
development exposure=`false`。

fit manifest、fold compatibility、exposure ledger 与 lineage manifest 均已写入同一 N4 run；lineage
共绑定 30 个 candidate/fold `.pt` artifact 与 3 个中间 JSON。审计确认每折 `4000/1000` fit/heldout
互斥，model q 为 `200×100`、float64、finite、nonnegative，且所有 artifact SHA 可复核。该结果
只证明当前 V2 development bank 上的 raw-q `R2_AMPLITUDE` compatibility；不证明 image/semantic/
optical/full-complex generation、training 或 readiness。

因此 N4 gate 是有效完成但无候选支持：`selected_producer=null`、`training_authorized=false`、
`formal_promotion=false`、`generation_readiness=not_ready`，sealed confirmation 仍为
`RESERVED_NOT_MATERIALIZED`/`consumed=false`。N4 当时未启动下一 package、sealed consumption、
training、promotion 或 generation；后续仅经独立 contract 与 Gate 执行了 fixed-R Student-t
comparison，结果仍未选出 producer，所有下游边界继续关闭。

## N3 fixed-R Student-t copula contract definition（2026-09-09）

在 N4 的 `N4_Q_PRODUCER_COMPARISON_NO_CANDIDATE_SUPPORTED` 边界之后，本轮仅按独立
definition-only 动作冻结唯一下一候选族
`EMPIRICAL_MARGINAL_FIXED_R_STUDENT_T_COPULA_V1`。合同 artifact 为
[`n3_empirical_marginal_fixed_r_student_t_copula_contract.v1.json`](../outputs/20260909_v2/runs/n3_empirical_marginal_fixed_r_student_t_copula_contract_v1_attempt_01/n3_empirical_marginal_fixed_r_student_t_copula_contract.v1.json)，
SHA-256=`0f18dccf35758e590255fac07436eb968c866ddc11a3e19df7b02e25770be1d7`，status/result=`valid_completed/READY`，
classification=`DEFINE_N3_EMPIRICAL_MARGINAL_FIXED_R_STUDENT_T_COPULA_CONTRACT_V1_READY`，claim=
`N3_EMPIRICAL_MARGINAL_FIXED_R_STUDENT_T_COPULA_MODEL_ASSUMPTION_DEFINED`。

该合同只改变 copula 径向族：保留 fit-only empirical marginals、稳定 tie/interpolation、
Gaussianized-score analytic-shrinkage 与同一个 fitting-derived `R_shrunk`，每个 fold 只新增一个
fit-only global `nu`。`nu` 在闭区间 `[2.01,1000000]` 内按固定 `R` 的 Student-t copula
pseudo-likelihood、`eta=log(nu-2)` deterministic bounded golden-section optimizer 拟合；
CDF、逆 CDF、float64 CPU、factorization、Gamma radial sampler、endpoint/nonfinite/failure semantics
均已冻结。未来若另行授权，必须仍用 V2 的 `5` folds、`4000/1000` fit/heldout、`200` model bank、
相同 raw-q biased ED、q95 与八个 guardrails，并严格要求 `5/5`；`4/5` 不支持，relative improvement
只作 descriptive。

定义 consumer 仅校验 JSON/manifest 与 byte hashes；q artifact 只作 digest boundary，未反序列化、
未读取 fit/heldout q、未运行 nu fit、sampling 或 evaluator，sealed confirmation 仍为
`RESERVED_NOT_MATERIALIZED`。该合同不构成 producer support 或 selection，当前
`selected_producer=null`、`training_authorized=false`、`formal_promotion=false`、
`generation_readiness=not_ready` 保持不变；未来 execution 需要新的独立 Gate。

## N3 fixed-R Student-t copula producer comparison（2026-09-09）

在上述 READY contract 与冻结的 V2 source/q/five-fold/ceiling authority 上，按新的独立
`EXECUTE_N3_EMPIRICAL_MARGINAL_FIXED_R_STUDENT_T_COPULA_V1` Gate 完成一次 CPU-only、
development-only q-producer comparison。唯一有效 report 为
[`n3_empirical_marginal_fixed_r_student_t_copula_producer_report.v1.json`](../outputs/20260909_v2/runs/n3_empirical_marginal_fixed_r_student_t_copula_producer_v1_attempt_01/n3_empirical_marginal_fixed_r_student_t_copula_producer_report.v1.json)，
SHA-256=`4a938b883d1a0cf3386b7e2a5f4c900a236cfed031364c4a1273ea3ebe6d77dd`；contract SHA
仍为 `0f18dccf35758e590255fac07436eb968c866ddc11a3e19df7b02e25770be1d7`。对应 lineage
manifest 为
[`n3_empirical_marginal_fixed_r_student_t_copula_lineage_manifest.v1.json`](../outputs/20260909_v2/runs/n3_empirical_marginal_fixed_r_student_t_copula_producer_v1_attempt_01/n3_empirical_marginal_fixed_r_student_t_copula_lineage_manifest.v1.json)，
SHA-256=`f2f943dab288726db6aaf3fb6c90acfe516ac7869fa9203a2c2a083731cf88da`。

Gate result=`VALID_COMPLETED`，classification=`N3_FIXED_R_STUDENT_T_COPULA_NOT_SUPPORTED`。
五个 fold 均 operationally valid，兼容 pattern=`[true,true,false,true,true]`、`4/5`；唯一失败
为 fold 2 的 `energy_distance`、`marginal_mean_l2` 与
`marginal_energy_distance_mean`，所以严格 `5/5` 规则不支持该 family。每折 fit-only global
`nu` 分别为 `118.35972638524892`、`118.04632498600743`、`117.03308605630197`、
`118.47406043505019`、`119.33122553612606`，均在冻结 `[2.01,1000000]` 域内；empirical
marginal 与 fitting-derived `R_shrunk` 保持不变。执行使用固定 V2 q95 与全部 8 个 raw-q
guardrails，每折一个 `200×100` model bank，总 evaluator exposure=`5/5`，无 adaptive exposure、
heldout tuning、seed retry、best-of-N 或 post-evaluation resampling。

相对 N4 Gaussian baseline 的 `4/5` pattern=`[true,true,true,true,false]`，Student-t 的失败
转移到 fold 2，N4 fold 4 的 `norm_energy_distance` 与 `squared_norm_energy_distance` 在
本次不再失败；该相对变化仅作 descriptive evidence，未参与 support 或 selection。该结果是
有效完成后的 scientific failure，不是 producer selection：`selected_producer=null`、
`training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready`，
sealed confirmation 仍为 `RESERVED_NOT_MATERIALIZED`/`consumed=false`；没有 CUDA、training、
optical forward、image/semantic generation 或 promotion。

## N3 q-producer compatibility self-null calibration（2026-09-09）

在 N4 Gaussian `4/5` 与 fixed-R Student-t `4/5` 的结果之后，先冻结了一次只包含这两个既有
producer 的 self-null contract，再执行一次完整的 model-generated calibration。contract 为
[`n3_q_producer_compatibility_self_null_calibration_contract.v1.json`](../outputs/20260909_v2/runs/n3_q_producer_compatibility_self_null_calibration_contract_v1_attempt_01/n3_q_producer_compatibility_self_null_calibration_contract.v1.json)，
SHA-256=`32ba12a8f09a850f0126bf528741928a247d957cf0a2105bd4329d102a97662c`；固定每个
producer/fold `20` trials（总 `200`），因为完整现有 evaluator 每个 trial 包含 `4` 个
`1000/200` DATA-vs-MODEL pair、`4` 个 `200/200` MODEL-vs-MODEL reference pair 和坐标级
ED，`200` trials/fold 不在 bounded CPU envelope 内。该较低 trial count 的 Monte Carlo
精度限制已在 observation 前冻结，不能用来支持小差异的概率结论。

唯一 report 为
[`n3_q_producer_compatibility_self_null_calibration_report.v1.json`](../outputs/20260909_v2/runs/n3_q_producer_compatibility_self_null_calibration_v1_attempt_01/n3_q_producer_compatibility_self_null_calibration_report.v1.json)，
SHA-256=`0e69db97c52cb6e72383d2cd24227a0e9937b4aafb88413af6a9216ce93860fb`，status/result=`valid_completed/VALID_COMPLETED`，
classification=`N3_Q_COMPATIBILITY_SELF_NULL_MIXED_OR_UNRESOLVED`。Gaussian 的五折完整
pass rate 为 `[0.80,0.90,1.00,0.90,0.90]`，implied five-fold product=`0.5832`；Student-t
为 `[0.95,0.95,0.90,1.00,0.95]`，product=`0.7716375`。两者 primary `energy_distance`
false-fail rate 均为 `0`，但完整八 guardrail 的 self-null failures 存在：主要集中在
`norm_energy_distance`、`squared_norm_energy_distance`，并出现 Gaussian fold 0 的
`marginal_mean_l2`、fold 1 的 `marginal_energy_distance_max` 与 Student-t fold 0 的
`marginal_mean_l2`。因此该 calibration 既不足以宣布 evaluator stable，也没有达到预注册的
false-fail limitation 类别；它只保留一个 mixed/unresolved 的 evaluator uncertainty。

该结果是 FACT：同一 fitted producer 独立生成 pseudo-DATA/MODEL/reference banks 后，当前
V2 mean-vs-model-reference-q95 evaluator 的有限模拟行为。MODEL_ASSUMPTION 仍未关闭：不能
据此把 N4/Student-t 的 isolated `4/5` failures 解释为纯 model mismatch，也不能把它们解释为
纯 evaluator noise。sealed confirmation 仍为 `RESERVED_NOT_MATERIALIZED`/`consumed=false`，
selected producer、training、formal promotion 与 generation readiness 均保持关闭；不执行
第三 producer family。按 mixed class，唯一下一动作是对这一个 self-null uncertainty 做一个
bounded decision；不进入 family zoo，也不重复 Gaussian/Student-t。

## N3 copula failure attribution measurement（2026-09-09）

`EXECUTE_N3_COPULA_FAILURE_ATTRIBUTION_V1` 已在 V2 completeness contract 下通过完整预检并
完成一次 CPU-only、development-only、read-only measurement。V2 contract SHA-256 为
`5c85c4e949e5850de337d776f6782bdc791493b333ab9dd7982269e9f214eb95`，base V1 SHA-256 为
`333b51abe2a6b1f3e96558273d4b6037267aa80678199d3bf6d0c300bb3508df`。唯一 machine report 为
[`n3_copula_failure_attribution_report.v1.json`](../outputs/20260909_v2/runs/n3_copula_failure_attribution_v1_attempt_01/n3_copula_failure_attribution_report.v1.json)，
SHA-256=`e9a834150baa006c619256b8306340e31593e1c9562a37ea46955b34ae86cac8`，Gate result 为
`VALID_COMPLETED`。

冻结原始 marginal `M` pattern 为 `fold 0/2/3=1`、`fold 1/4=0`；V2 self-rank Gaussianized
primary dependence 在五个 fold 均为 `D=1`。因此 fold labels 为
`0=MARGINAL_AND_DEPENDENCE`、`1=DEPENDENCE_ONLY`、`2=MARGINAL_AND_DEPENDENCE`、
`3=MARGINAL_AND_DEPENDENCE`、`4=DEPENDENCE_ONLY`，精确 aggregate 为
`N3_COPULA_FAILURE_MARGINAL_AND_DEPENDENCE_MIXED`。两个原始 pass-control folds 也出现
`DEPENDENCE_ONLY`，按 V2 保守 aggregate mapping 不能归入 marginal-only 或 dependence-dominant。

该结果只消费冻结的 60/60 historical generated-bank identities、原始五折 assignment 与
R2 amplitude development pool；fit/held-out 为每 fold `800/200`，不读取 sealed confirmation。
Spearman、spectral、support/tail 均按 V2 保持 descriptive-only，不能改写 `M`、`D` 或最终
classification。报告明确记录了固定 development pool 的 fold reuse limitation；没有执行 P1、
producer fit/sampling、selector V3、CUDA、optimizer/backward/training、complex OLS 或 generation
promotion。`training_authorized=false`、`formal_promotion=false`、
`generation_readiness=not_ready` 与 sealed confirmation `RESERVED_NOT_MATERIALIZED`、
`consumed=false` 保持不变。

## N3 raw-q mixed component-swap diagnostic contract definition（2026-09-09）

`DEFINE_N3_COPULA_MIXED_COMPONENT_SWAP_DIAGNOSTIC_CONTRACT_V1` 已完成 definition-only
authority 校验，classification=`DEFINE_N3_COPULA_MIXED_COMPONENT_SWAP_DIAGNOSTIC_CONTRACT_V1_READY`，
claim=`N3_COPULA_MIXED_COMPONENT_SWAP_DIAGNOSTIC_CONTRACT_DEFINED`。权威 artifact 为
[`n3_copula_mixed_component_swap_diagnostic_contract.v1.json`](../outputs/20260909_v2/runs/n3_copula_mixed_component_swap_diagnostic_contract_v1_attempt_01/n3_copula_mixed_component_swap_diagnostic_contract.v1.json)，
SHA-256=`d46c4d8e8d8dbc41f325f2e1d7f001d645479a60deef08e5d514a80bada22763`。

该 contract 固定五折历史 pairing：每折 held-out DATA bank 对应四个 frozen
DATA-vs-MODEL banks，MODEL-vs-MODEL q95 只使用原始四个 reference pairs；ordinal rank 的
数值升序与 exact-tie 原始 row index 次序、`H_MARG[i,j]=sorted_D[r_M(i,j),j]`、
`H_DEP[i,j]=sorted_M[r_D(i,j),j]`、model/data self-reconstruction identity、pair/fold/aggregate
oracle-sufficiency mapping 均已 machine-defined。主指标严格复用原始 raw q-space biased empirical
energy distance 与原始 fold-local q95，不把 fold-level compatibility 误作四个 pair 全部失败。

本次只读取 JSON/manifest 与 byte hashes，未加载 q tensor、未构造 hybrid、未计算
`E_BASE`/`E_MARG`/`E_DEP`，contract-totality=`passed`；没有 producer fit/sampling、sealed
confirmation、CUDA、optimizer/backward/training、complex OLS、N4/N5/N6 或 generation action。
该 contract 不是 producer-support classification，也不授权后续执行。既有 attribution report
仍为 `N3_COPULA_FAILURE_MARGINAL_AND_DEPENDENCE_MIXED`，copula producer 仍为
`N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_NOT_SUPPORTED`；`training_authorized=false`、
`formal_promotion=false`、`generation_readiness=not_ready` 与 sealed
`RESERVED_NOT_MATERIALIZED`/`consumed=false` 不变。

## N3 raw-q mixed component-swap diagnostic execution（2026-09-09）

`EXECUTE_N3_COPULA_MIXED_COMPONENT_SWAP_DIAGNOSTIC_V1` 已在冻结 component-swap contract 下
通过完整 static preflight，并完成唯一一次 CPU-only、development-only、read-only diagnostic。
Contract SHA-256=`d46c4d8e8d8dbc41f325f2e1d7f001d645479a60deef08e5d514a80bada22763`；唯一 machine
report 为
[`n3_copula_mixed_component_swap_diagnostic_report.v1.json`](../outputs/20260909_v2/runs/n3_copula_mixed_component_swap_diagnostic_v1_attempt_01/n3_copula_mixed_component_swap_diagnostic_report.v1.json)，
SHA-256=`fc139cb6ec9754babeb9e4ac6c03f08ef339cbdd6148362bb05b44c08daeaecc`，Gate result 为
`VALID_COMPLETED`。

精确 aggregate classification=`N3_COPULA_PRIMARY_BOTH_ORACLES_SUFFICIENT`。五折 labels 为
`0/1/2/3=BOTH_ORACLES`、`4=CONTROL_FOLD`；20 个历史 DATA-vs-MODEL comparisons 均输出
pair label，其中原始 primary-failing pairs 为 `9/9 BOTH_ORACLES`，原始 passing controls 为
`11/11 BOTH_ORACLES`，controls 只作 descriptive 输出，不进入 sufficiency aggregate。每个
fold-local q95 原样复用原始 producer report，未用 oracle hybrid 重算阈值。

该消费器验证了 `5/5` held-out DATA banks 与 `60/60` MODEL banks 的 ordinal rank
self-reconstruction、每坐标 permutation、exact-tie row-order、finite/nonnegative 与无 RNG
identity；20/20 的 `E_BASE` 与原始 report 字段精确相等。所有 `H_MARG`/`H_DEP` 明确标注为
`DIAGNOSTIC_ONLY`、`ORACLE_USES_HELDOUT_DATA`、`NOT_A_PRODUCER`、
`NOT_GENERATION_EVIDENCE`。没有 producer fit/sampling、bank rematerialization、sealed
confirmation、P1 execution、CUDA、optimizer/backward/training、complex OLS、N4/N5/N6 或
generation promotion；`p1_execution_authorized=false`、`training_authorized=false`、
`formal_promotion=false`、`generation_readiness=not_ready` 保持不变。该结果只支持一个理想化
held-out-data component intervention 在冻结 raw primary metric 下的 oracle sufficiency，不
支持可学习 producer、generalization、confirmation 或 generation readiness。

## N3 oracle-guided minimal intervention contract definition（2026-09-09）

`DEFINE_N3_ORACLE_GUIDED_MINIMAL_INTERVENTION_CONTRACT_V1` 已完成 definition-only authority
校验，classification=`DEFINE_N3_ORACLE_GUIDED_MINIMAL_INTERVENTION_CONTRACT_V1_READY`，
claim=`N3_ORACLE_GUIDED_MINIMAL_INTERVENTION_CONTRACT_DEFINED`。唯一 machine contract 为
[`n3_oracle_guided_minimal_intervention_contract.v1.json`](../outputs/20260909_v2/runs/n3_oracle_guided_minimal_intervention_contract_v1_attempt_01/n3_oracle_guided_minimal_intervention_contract.v1.json)，
SHA-256=`d640335c298afed696c64250acd15ed4767e8222a8e962f54fe6cd05c8a0f6b1`。

该 contract 只冻结两个且仅两个 development-only 候选：Candidate M 为逐坐标
`log1p(q)` Gaussian location-scale marginal（`mu_j`、`sigma_j`，`n-1` 分母、解析
`expm1` inverse quantile），完整保留当前 Gaussianized-score analytic-shrinkage
`R_shrunk`；Candidate D 为固定 `nu=5` 的 Student-t copula，使用同一 `R_shrunk`、一个
全局径向 scale 与明确的 `F_t5` probability-integral transform，完整保留当前 fit-only
empirical marginal plotting-position/quantile 规则。两者都无 optimizer、无 label conditioning、
无 mixture/grid/flow/neural path；任何 finite-domain、strict-probability 或 output failure
均 fail-closed 为 `INVALID_HARNESS`。

复杂度在执行前冻结为：M 的有效 fitted scalar count=`5250`（`5050` 个唯一对称
`R_shrunk` entries + `200` 个 marginal scalars），D=`85050`（`5050` + `800×100`
个 empirical order-statistic slots）；lexicographic ordering 因此在双方通过时预先选择 M，
最终残余 tie 仍固定 M before D。后续比较只允许一次五折 `800/200`、固定 assignment
SHA=`90277c16a0d96b895e218264f35a822564b3a46570cab73a1a1e823f063e9e32`、每候选每折
`4/8` banks，raw q-space ED 与当前 8 个标量 compatibility guardrails，且必须 5/5
fold 全部通过；不使用相对改善或加权 aggregate score。

本次未拟合、未采样、未读取任何新候选 q/result；oracle、attribution、current copula、
sealed confirmation 与全局 flags 均不变：`N3_COPULA_PRIMARY_BOTH_ORACLES_SUFFICIENT`、
`N3_COPULA_FAILURE_MARGINAL_AND_DEPENDENCE_MIXED`、
`N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_NOT_SUPPORTED`、sealed
`RESERVED_NOT_MATERIALIZED`/`consumed=false`、`training_authorized=false`、
`formal_promotion=false`、`generation_readiness=not_ready`。该定义不授权比较执行、
producer freeze、confirmation、N4/N5/N6、training 或 generation；Sources/bridge 仍未 refresh。

## N3 oracle-guided minimal intervention comparison blocked（2026-09-09）

`EXECUTE_N3_ORACLE_GUIDED_MINIMAL_INTERVENTION_COMPARISON_V1` 已完成静态契约、上游 authority、
五折身份与数据 lineage 校验，并按冻结 contract 进入 CPU-only 执行；未调用 CUDA、optimizer、
backward 或 training。唯一 machine report 为
[`n3_oracle_guided_minimal_intervention_comparison_report.v1.json`](../outputs/20260909_v2/runs/n3_oracle_guided_minimal_intervention_comparison_v1_attempt_01/n3_oracle_guided_minimal_intervention_comparison_report.v1.json)，
SHA-256=`ff8404e4c6b26b66d9292e728e30031cc5b14b7c1d1396d5713e6a2bdae77c17`，Gate result 为
`BLOCKED`，scientific comparison classification 与 selected candidate 均为 `null`。

阻塞发生在 Candidate M 的第 `0` 折、`DATA-vs-MODEL`、replicate `0`：冻结的
`Q_M(u)=expm1(mu+sigma*Phi^-1(u))` 产生 `351/20000` 个负输出，`q_min=-0.42263710721584474`、
`q_max=7.528838701593836`，没有 non-finite 值。Contract 明确要求 `expm1` 输出 finite/nonnegative，
禁止 clamp、abs、ReLU、softplus、epsilon repair 与 proposal rejection，并规定此情形为
`INVALID_HARNESS`，在兼容性、5/5 support 与 candidate selection 之前停止。因此没有将该结果
误写为 `NO_MINIMAL_INTERVENTION_SUPPORTED`，Candidate D 也未继续执行；没有 partial candidate
selection 或任何科学兼容性结论。

该阻塞证明当前 `log1p` marginal 分支在冻结非负输出边界下不可执行，但不授权修改 contract、
定义 V2、消费 sealed confirmation、final all-development fit、P1、N4/N5/N6、producer promotion
或 generation。`N3_COPULA_PRIMARY_BOTH_ORACLES_SUFFICIENT`、
`N3_COPULA_FAILURE_MARGINAL_AND_DEPENDENCE_MIXED` 与原始 copula classification 保持不变；
 sealed confirmation 仍为 `RESERVED_NOT_MATERIALIZED`/`consumed=false`，
`training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready` 保持不变。

## N3 oracle-guided minimal intervention contract V2 definition（2026-09-09）

针对 V1 Candidate M 的 `INVALID_HARNESS` 支持阻塞，`DEFINE_N3_ORACLE_GUIDED_MINIMAL_INTERVENTION_CONTRACT_V2`
已完成一次 definition-only、CPU-only contract consumer。V1 contract 保持 immutable，V1
invalid-harness report 也保持 immutable；V2 只把 Candidate M 替换为
`ZERO_TRUNCATED_LOG1P_GAUSSIAN_MARGINAL_V1`（Candidate M2），没有读取新的 development
outcome、q tensor 或 sealed confirmation，也没有拟合/采样 M2 或 D。

V2 machine authority 为
[`n3_oracle_guided_minimal_intervention_contract.v2.json`](../outputs/20260909_v2/runs/n3_oracle_guided_minimal_intervention_contract_v2_attempt_01/n3_oracle_guided_minimal_intervention_contract.v2.json)，
SHA-256=`f297a1ffbf84bec84a15525be97ba1a5fd897d32beefd0fe22138bff0012ce2e`，classification 为
`DEFINE_N3_ORACLE_GUIDED_MINIMAL_INTERVENTION_CONTRACT_V2_READY`，claim 为
`N3_ORACLE_GUIDED_MINIMAL_INTERVENTION_CONTRACT_V2_DEFINED`。它绑定 V1 contract SHA
`d640335c298afed696c64250acd15ed4767e8222a8e962f54fe6cd05c8a0f6b1` 与 V1
`INVALID_HARNESS` report SHA `ff8404e4c6b26b66d9292e728e30031cc5b14b7c1d1396d5713e6a2bdae77c17`。

M2 保留 V1 的 `mu_j`/`sigma_j` 拟合、Gaussian copula dependence、RNG namespace 与所有
comparison 语义，仅将 inverse marginal 固定为
`alpha_j=Phi(-mu_j/sigma_j)`、`p_j=alpha_j+(1-alpha_j)u_j`、
`y_j=mu_j+sigma_j*Phi^-1(p_j)`、`q_j=expm1(y_j)`。解析证明为
`p_j>=alpha_j`、`Phi^-1` 单调且 `y_j>=0`，故 `q_j>=0` by construction。浮点边界已冻结为
`q_raw`/`y_raw` 绝对容差 `1e-12`：仅在两者均处于该零边界内时 canonicalize 到 `q=0.0`，
material negative 仍为 `INVALID_HARNESS`，不使用 hidden clamp/rejection/repair。

V2 的 5 个 deterministic synthetic support cases、contract-totality 与 semantic-diff 均通过；
Candidate D 为 byte/semantic-equivalent，复杂度仍为 M2=`5250`、D=`85050`，双方通过时
仍只因 `5250 < 85050` 选择 M2。该定义完成后，新的 M2-vs-D 五折 development comparison
已按独立授权执行一次，结果见下方 V2 execution 条目；
不授权 final all-development fit、sealed confirmation、P1、N4/N5/N6、training 或 generation；
全局 flags 与 sealed `RESERVED_NOT_MATERIALIZED`/`consumed=false` 保持不变。

## N3 oracle-guided minimal intervention comparison V2 execution（2026-09-09）

`EXECUTE_N3_ORACLE_GUIDED_MINIMAL_INTERVENTION_COMPARISON_V2` 已在 V2 contract
`f297a1ffbf84bec84a15525be97ba1a5fd897d32beefd0fe22138bff0012ce2e` 下完成完整 preflight，
并从 fold `0` 到 `4` 对 Candidate M2 与 Candidate D 各新鲜执行一次 fit、4 个
DATA-vs-MODEL banks、8 个 MODEL-vs-MODEL banks 与 raw-q ED/8 个 compatibility guardrails。
唯一 machine report 为
[`n3_oracle_guided_minimal_intervention_comparison_report.v2.json`](../outputs/20260909_v2/runs/n3_oracle_guided_minimal_intervention_comparison_v2_attempt_01/n3_oracle_guided_minimal_intervention_comparison_report.v2.json)，
SHA-256=`19d4d933f28200010373ed8210f31f343ae40d6474f6bdfb05e7fcb0c5fcf192`，Gate result 为
`VALID_COMPLETED`。

M2 的五个 fold 均 operationally valid，support 检查为 finite/nonnegative，materially-negative
输出=`0`、roundoff canonicalization=`0`，但完整 compatibility 为 `0/5`；fold 3 虽通过四个
joint guardrails，仍未通过两个 aggregate marginal guardrails。D 的五个 fold 也均
operationally valid，固定 `nu=5` 在全部 fold 验证通过，但仅 fold 3 完整通过八个 guardrails，
compatibility 为 `1/5`。因此两候选均未达到冻结的 `5/5` development support，精确分类为
`NO_MINIMAL_INTERVENTION_SUPPORTED`，`selected_candidate=null`；没有使用 relative improvement
或 weighted score。

该结果只支持当前固定 1000-row development pool 上的 development-only 比较，不支持
producer family、generalization、population truth、confirmation 或 generation。V1 的历史
`INVALID_HARNESS` report 保持 immutable；本次未执行 final all-1000 fit、sealed confirmation、
complex OLS、N4/N5/N6、CUDA、optimizer、backward、training 或 generation promotion。全局 flags
仍为 `training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready`，
sealed confirmation 仍为 `RESERVED_NOT_MATERIALIZED`/`consumed=false`；Sources/bridge 尚未 refresh，
不声明 publication 为 `current_clean`。

## N3 empirical-fit joint bootstrap ceiling diagnostic（2026-09-09）

`DEFINE_N3_EMPIRICAL_FIT_JOINT_BOOTSTRAP_CEILING_DIAGNOSTIC_V1` 已先完成 definition-only
authority、fold identity、compatibility applicability 与 contract-totality 校验；合同定义本身
未加载 q。唯一 machine contract 为
[`n3_empirical_fit_joint_bootstrap_ceiling_diagnostic_contract.v1.json`](../outputs/20260909_v2/runs/n3_empirical_fit_joint_bootstrap_ceiling_diagnostic_v1_attempt_01/n3_empirical_fit_joint_bootstrap_ceiling_diagnostic_contract.v1.json)，
SHA-256=`fc24ca4c863fded5f23dacc69b13bb16f44224b96d47888d62cb9a3f0f70a38d`。

`N3_EMPIRICAL_FIT_JOINT_BOOTSTRAP_CEILING_DIAGNOSTIC_V1` 随后按该 immutable contract 完成唯一
一次 CPU-only、development-only execution。每折使用 exact `800/200` fit/held-out rows，
从 fitting q rows 中有放回抽取完整 q 行，生成 `4` 个 DATA-vs-MODEL 与 `8` 个
MODEL-vs-MODEL banks；没有独立维度抽样、seed retry 或 best-bank selection。8 个既有标量
guardrails 均被合同预先判定为适用，per-coordinate marginal vector 与 bootstrap duplicate-row
rate 仅为 descriptive。

唯一 machine report 为
[`n3_empirical_fit_joint_bootstrap_ceiling_diagnostic_report.v1.json`](../outputs/20260909_v2/runs/n3_empirical_fit_joint_bootstrap_ceiling_diagnostic_execution_v1_attempt_01/n3_empirical_fit_joint_bootstrap_ceiling_diagnostic_report.v1.json)，
SHA-256=`c4557d44c8d1b40adeb2f362a4f81f0f28d9feb398428fbb1b13c035b043e12c`，Gate result=`VALID_COMPLETED`，
精确 classification=`N3_FIT_HELDOUT_EMPIRICAL_CEILING_NOT_SUPPORTED`。五折均 operationally
valid；fold 0/1/2/3/4 的 primary `DATA-vs-MODEL mean ED` 对 `MODEL-vs-MODEL q95` 分别为
`0.0697057720/0.0797288600`、`0.0559347173/0.0898967185`、
`0.0662504869/0.0952121402`、`0.0689992654/0.0913000623`、
`0.0583398880/0.0802141875`，完整 compatibility 分别为
`false/true/false/true/true`。失败 fold 0 仅为 `marginal_std_l2`，fold 2 仅为
`marginal_energy_distance_max`；每个 bootstrap bank 的 duplicate-row rate 只作描述，五折
所有 12 banks 的折内平均约为 `0.1071/0.1146/0.1054/0.1129/0.1150`。

因此 empirical fitting joint resampling 通过 `3/5` 而未达到冻结的 `5/5` development criterion。
FACT 是该 ceiling 在当前固定 development pool 上未通过；MODEL_ASSUMPTION 是结果与
fit-to-heldout distribution stability / finite-development sampling limitation 一致，而不是
对 population-level producer 不存在的证明。该诊断不是 producer、novel generation 或
confirmation evidence；未执行 M3/新 copula/KDE/mixture/vine/flow/neural density、producer
selection、final all-1000 fit、sealed confirmation、complex OLS、N4/N5/N6、CUDA 或 training。
`training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready` 与 sealed
`RESERVED_NOT_MATERIALIZED`/`consumed=false` 保持不变；下一规划目标为 development
sampling/stationarity contract，不升级 producer family。

## N3 development-sampling stationarity diagnostic（2026-09-09）

在 empirical-fit ceiling diagnostic 之后，按独立 bounded authorization 定义并执行一次
`N3_DEVELOPMENT_SAMPLING_STATIONARITY_DIAGNOSTIC_V1`，只消费 frozen 1000-row N3
development bank、原始五折 assignment、已注册 source-position authority 与既有 empirical-ceiling
fold linkage。Contract definition-only 阶段未加载 q，唯一 machine contract 为
[`n3_development_sampling_stationarity_diagnostic_contract.v1.json`](../outputs/20260909_v2/runs/n3_development_sampling_stationarity_diagnostic_v1_attempt_01/n3_development_sampling_stationarity_diagnostic_contract.v1.json)，
SHA-256=`05c0c5cb318566d59e4d46da9693cede0cbf9958b69656510b6deaf3e6e18380`，
contract-totality=`passed`。

Contract 固定 raw-q 的 coordinate-mean L2、coordinate-std L2 与 row-radial moment L2 三个
fit-vs-heldout scalar；固定 1000 个 balanced random-partition references，每个 reference 保留
五个 `80/20`-per-class pseudo-fold，global q95、class-conditioned q99，CPU deterministic seed
schedule。官方五折未重分、未重选、未 pooling；class composition 始终为每类 `80/20`，不能解释
结果。Source-position 复用既有 within-class ascending `source_index` convention；fold subset
raw-q energy 只作 descriptive，不创建新 selector。

首次 execution attempt 因直接统计实现的浮点负方差在科学分类前以 `INVALID_HARNESS` blocked，
该 write-once report 保留且没有 scientific classification；修复为直接 subset moment computation
后，未改变 contract、阈值、随机分区或分类语义。唯一有效 report 为
[`n3_development_sampling_stationarity_diagnostic_report.v1.json`](../outputs/20260909_v2/runs/n3_development_sampling_stationarity_diagnostic_execution_v1_attempt_02/n3_development_sampling_stationarity_diagnostic_report.v1.json)，
SHA-256=`625b4edda39066a4c414337fc0ba7d06d64fda77b6d41d8241806c715ee871a3`，Gate result=`VALID_COMPLETED`，
classification=`N3_DEVELOPMENT_SAMPLING_STABILITY_LIMITATION_SUPPORTED`。

五折 global moment 都落在 frozen finite-partition reference 内：fold 0/1/2/3/4 的
`coordinate_mean_l2` 为 `0.333390/0.199075/0.407468/0.297724/0.239533`，
`coordinate_std_l2` 为 `0.215446/0.176665/0.266582/0.252756/0.259172`，
`radial_moment_l2` 为 `0.111403/0.028738/0.105054/0.167771/0.099436`，
对应 reference percentiles 分别为 `0.6734/0.0056/0.9090/0.4584/0.0898`、
`0.1944/0.0076/0.7072/0.5828/0.6438`、`0.4526/0.0412/0.4190/0.7300/0.3852`；因此
`exceptional_folds=0`，fold 0/2 不被该有限随机分区 reference 判为 statistically exceptional。
Class-conditioned q99 仅 fold 1 的 label 6 为 exceptional，未达到冻结的 class-drift rule；所有
fold 的 composition difference=`false`。既有 empirical-ceiling linkage 原样保留为
`false/true/false/true/true`，没有重新计算或改写其 `3/5` classification。

限制性 classification 来自已注册 source-position authority：pooled class-stratified r2
Spearman=`0.4472886`，超过冻结阈值 `0.25`，且 `10/10` classes 的 r2 half-difference 与
Spearman signs 均为 positive。FACT 是五折 moment 差异与有限分区 reference 相容、而 registered
source-position drift 被支持；MODEL_ASSUMPTION 是应先修复 development sampling/stationarity
protocol，不能据此证明 population exchangeability、producer approximation gap 或 folds 0/2 的
唯一原因。该 diagnostic 不是 producer、producer selection、confirmation 或 generation evidence；
`training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready` 与 sealed
`RESERVED_NOT_MATERIALIZED`/`consumed=false` 保持不变。唯一下一动作是停止 producer-family escalation，
先 redefine/repair development sampling/stationarity protocol。

## N3 source-position-balanced development protocol and empirical ceiling（2026-09-09）

在 stationarity limitation 之后，按单一 bounded authorization 定义并冻结
`N3_SOURCE_POSITION_BALANCED_DEVELOPMENT_PROTOCOL_V1`。唯一 protocol artifact 为
[`n3_source_position_balanced_development_protocol.v1.json`](../outputs/20260909_v2/runs/n3_source_position_balanced_development_protocol_v1_attempt_01/n3_source_position_balanced_development_protocol.v1.json)，
SHA-256=`5700ebb3b4b163d6d911763ce70ef77ca85f1c3ef387b7b0fdaed153e36e0904`，
row→fold assignment SHA=`07d91587683792ba307f35dc06761ef3c55be49c52f05d0ddab0c3912d54e4d8`。
定义阶段只读取既有 row/class/source-position metadata，`q_values_used=false`；旧五折
`90277c16a0d96b895e218264f35a822564b3a46570cab73a1a1e823f063e9e32` 保持 immutable。

新 protocol 对每类按 frozen `source_index` ascending 排序，tie 以原始 development-row identity
ascending，随后 `heldout_fold = rank mod 5`。五折均为每类 `20` heldout、总计 `200` heldout / `800`
fit，且每类覆盖 rank `0..99` 的固定 round-robin residues；无 RNG、无候选 split、无 split optimization。

随后仅执行一次 `N3_SOURCE_POSITION_BALANCED_EMPIRICAL_CEILING_V1`，复用旧 empirical ceiling 的
complete-row bootstrap、4/8 banks、raw-q ED、8 项 guardrails、fold-local q95 与严格 `5/5` 规则。
唯一 report 为
[`n3_source_position_balanced_empirical_ceiling_report.v1.json`](../outputs/20260909_v2/runs/n3_source_position_balanced_empirical_ceiling_v1_attempt_01/n3_source_position_balanced_empirical_ceiling_report.v1.json)，
SHA-256=`f9f86842aa063e67264a5be54a242eb37e9403756c9001b1f8b9f5e653dfd989`，status=`valid_completed`、
result=`VALID_COMPLETED`，classification=`SOURCE_POSITION_BALANCED_DEVELOPMENT_PROTOCOL_NOT_SUPPORTED`，
compatible=`0/5`。各 fold 的 raw-q ED/q95 为 `0.056833/0.070078`、`0.055111/0.072053`、
`0.076349/0.068817`、`0.076026/0.073709`、`0.077991/0.090075`；失败 guardrail 分别为
fold 0=`marginal_energy_distance_max`，fold 1=`covariance_relative_frobenius,marginal_std_l2`，
fold 2=`energy_distance,marginal_energy_distance_mean,marginal_energy_distance_max,marginal_mean_l2,norm_energy_distance,squared_norm_energy_distance`，
fold 3=`energy_distance,marginal_energy_distance_mean,marginal_mean_l2,marginal_std_l2,norm_energy_distance,squared_norm_energy_distance`，
fold 4=`marginal_std_l2`。五折均 operationally valid；duplicate-row rate 仅 descriptive，折内 12 个
bank 的平均约为 `0.1071/0.1146/0.1054/0.1129/0.1150`。

FACT 是 source-position balancing 在当前 1000-row bank 上没有恢复任何一折的完整 empirical
fit-to-heldout compatibility；MODEL_ASSUMPTION 是仅平衡 observed source-position 仍不足以形成
稳定 development authority，不代表不存在 population-level producer，也不重解释旧 `3/5` evidence。
未评估 producer、M2、Student-t、KDE/flow/mixture、alternate split、sealed confirmation、CUDA、
training、generation 或 N4/N5/N6；`selected_producer=null`、`training_authorized=false`、
`formal_promotion=false`、`generation_readiness=not_ready` 保持不变。下一动作是停止当前 1000-row
bank 上的 producer-family escalation，规划 replacement/expanded N3 development-data authority。

最新 N3 local-shrinkage-covariance development selection 为
`N3_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_INCONCLUSIVE`，没有 selected K。Stage-1 的 15/15
candidate-fold 均收敛；Stage-2 因 `n_k<2` 在 K=2/4/8 分别有 1/2/4 个 fold
`LOCAL_COVARIANCE_COMPONENT_INSUFFICIENT`，其余 operational folds 均未通过 frozen
primary/covariance/norm compatibility。该结果是 valid development-only inconclusive，不能
解释为 covariance heterogeneity supported、K selected 或 producer family established。

本文件后续带日期或明确标为历史的段落是不可覆盖的历史事实；其中的“下一步”只表示当时的状态，
不覆盖本摘要。详细数值与输入绑定由对应 machine evidence 和 experiment history 维护。

## N3 sealed confirmation source reservation（2026-09-08）

`DEFINE_N3_SEALED_CONFIRMATION_SOURCE_V1` 已完成，classification=
`N3_SEALED_CONFIRMATION_SOURCE_RESERVED`。Manifest 位于
[`n3_sealed_confirmation_source_manifest.v1.json`](../outputs/20260908_v2/runs/n3_sealed_confirmation_source_v1_attempt_01/n3_sealed_confirmation_source_manifest.v1.json)，
状态为 `RESERVED_NOT_MATERIALIZED`，manifest SHA-256 为
`6475474f157d31591a3b0252f27320323d84ea93803c61ee6a365420a8ebb08e`。

这是 fresh balanced confirmation source reservation（500 rows、50/class、适用 overlap=0），
不是 OLS、producer、rank/family、selector 或 generation 结果。它只允许未来已冻结 N3
producer confirmation consumer 一次性消费；完整选择规则和 firewall 输入以 machine manifest
为准，日期化 provenance 见 [experiment history](status/experiment_history.md)。

## N3 minimal competing producer baselines（2026-09-08）

`COMPARE_N3_MINIMAL_COMPETING_PRODUCER_BASELINES_V1` 在既有 1000-row N3 development
pool 上完成了固定、output-independent 的 5-fold comparison；没有读取或 materialize
sealed confirmation source。新的 fold assignment SHA 为
`90277c16a0d96b895e218264f35a822564b3a46570cab73a1a1e823f063e9e32`。Candidate A
`SHRINKAGE_ELLIPTICAL_GAUSSIAN_AMPLITUDE_V1` 与 Candidate B
`RADIAL_SCALE_MIXTURE_AMPLITUDE_V1` 均在 5/5 folds operationally valid，固定 positivity
budget 下每个 data-vs-model 与 model-vs-model bank 均达到 200 accepted；但每个 fold 的
primary q-space energy discrepancy 都超出各自 model-vs-model q95，covariance/norm
guardrails 也未通过。因此结果为
`N3_MINIMAL_PRODUCER_BASELINES_NOT_SUPPORTED`，没有冻结候选。B 的 aggregate primary
较 A 低（约 `0.5456` vs `0.7404`），但仍不兼容，不能因相对较好而选择。

权威 report 为
[`n3_minimal_competing_producer_baselines_report.v1.json`](../outputs/20260908_v2/runs/n3_minimal_competing_producer_baselines_v1_attempt_01/n3_minimal_competing_producer_baselines_report.v1.json)，
SHA-256=`af7611cfbca211e666d7773e81b230a1b380b91583817c3c4ab80925f2a16440`。
该结果仅是 development comparison；不支持新 producer family、independent confirmation、
full complex OLS distribution、semantic generation 或 generation readiness。`training_authorized=false`、
`formal_promotion=false` 与 `generation_readiness=not_ready` 保持；sealed reservation 仍不授权
消费。后续若继续 producer research，必须另立最小、独立的 research question 与 authority，
不得把本次失败变成调参、第二个 canary 或 selector V3 授权。

## N3 unsupervised local location mixture contract（2026-09-08）

`DEFINE_N3_UNSUPERVISED_LOCAL_LOCATION_MIXTURE_PRODUCER_CONTRACT_V1` 已完成，
classification=`DEFINE_N3_UNSUPERVISED_LOCAL_LOCATION_MIXTURE_PRODUCER_CONTRACT_V1_READY`。
Contract 固定一个无条件、无标签的 shared-covariance location mixture，候选
`K={2,4,8}`；共享 covariance 复用上一轮已实现并测试的 analytic shrinkage convention，
初始化、CPU float64 EM、component canonicalization、collapse/failure、positivity sampling、
model-vs-model compatibility 和 smallest-K selection 均已冻结。它复用上一轮 5-fold assignment
SHA=`90277c16a0d96b895e218264f35a822564b3a46570cab73a1a1e823f063e9e32`。
权威 contract 为
[`n3_unsupervised_local_location_mixture_producer_contract.v1.json`](../outputs/20260908_v2/runs/n3_unsupervised_local_location_mixture_producer_contract_v1_attempt_01/n3_unsupervised_local_location_mixture_producer_contract.v1.json)，
SHA-256=`38cb0c549d8f943123f0ab3304adf5a01e9b3b5edb63daa384c7f4c7bac20449`。
Contract-definition phase 只允许 claim=`N3_LOCAL_LOCATION_MIXTURE_MODEL_ASSUMPTION_DEFINED`；该阶段
没有执行真实 N3 mixture fit、K selection、sampling 或 confirmation。随后执行结果见下节；
`training_authorized=false`、`formal_promotion=false`、`generation_readiness=not_ready` 不变。

## N3 local-location mixture development selection（2026-09-08）

`EXECUTE_N3_LOCAL_LOCATION_MIXTURE_SELECTION_V1` 严格消费上述 frozen contract，在同一 5-fold
development split 上执行了 `K={2,4,8}`。三个 K 均为 5/5 folds operationally valid；每个 fold 的
CPU float64 EM 均收敛，positivity-conditioned data/model banks 均达到固定 target，未发生
component collapse。可是每个 K 的 5/5 folds 都未同时通过 primary q-space energy 与 covariance/norm
compatibility guardrails，因此没有 candidate compatible，结果为
`N3_LOCAL_LOCATION_MIXTURE_NOT_SUPPORTED`，`K_selected=null`。

各 K 的 aggregate primary / covariance / norm-energy guardrail summary 分别为：K=2
`0.724299 / 0.794114 / 12.014982`，K=4 `0.757225 / 0.662969 / 13.119141`，K=8
`0.730527 / 0.677692 / 12.682548`；相对较小的单项数值不构成 compatibility 或模型选择。
权威 report 为
[`n3_local_location_mixture_selection_report.v1.json`](../outputs/20260908_v2/runs/n3_local_location_mixture_selection_v1_attempt_01/n3_local_location_mixture_selection_report.v1.json)，
SHA-256=`5064ee87cbc9846f0194ba0afcea4687647ceefb454d2abb20b8b56d632a9080`。
该结果是 development-only valid failure；没有执行 final all-1000-row producer fit、sealed
confirmation、independent confirmation、novel OLS 或 generation。

## N3 unsupervised local shrinkage-covariance mixture contract（2026-09-08）

在 `N3_LOCAL_LOCATION_MIXTURE_NOT_SUPPORTED` 之后，
`DEFINE_N3_UNSUPERVISED_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_CONTRACT_V1` 已完成并冻结一个
只改变 local covariance geometry 的下一 family。它保留上一 contract 的无监督 local-location
partition、`K={2,4,8}`、5-fold identity、positivity sampling 和 model-vs-model compatibility；
Stage-1 仍使用既有 shared-covariance local mixture，Stage-2 对 hard-assigned fitting rows
分别使用同一 analytic shrinkage convention 构造 `Sigma_k`，`n_k<2` 为
`LOCAL_COVARIANCE_COMPONENT_INSUFFICIENT`，不得 merge/restart/change K，也不运行第二次 EM。

权威 contract 为
[`n3_unsupervised_local_shrinkage_covariance_mixture_producer_contract.v1.json`](../outputs/20260908_v2/runs/n3_unsupervised_local_shrinkage_covariance_mixture_producer_contract_v1_attempt_01/n3_unsupervised_local_shrinkage_covariance_mixture_producer_contract.v1.json)，
SHA-256=`2cb596abe61deeaa847b90c74654c5abe0e94aae18a6f252170f8149ad1c1550`。
当前只允许 claim=`N3_LOCAL_COVARIANCE_HETEROGENEITY_MODEL_ASSUMPTION_DEFINED`；没有执行
local-covariance fit、K selection、sampling、sealed confirmation 或 generation。

## N3 local shrinkage-covariance mixture development selection（2026-09-08）

`EXECUTE_N3_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_SELECTION_V1` 严格消费上述 frozen contract，
只使用同一 1000-row development pool、5-fold assignment 和 CPU float64 Stage-1/Stage-2
规则。K=2/4/8 的 operational folds 分别为 4/3/1；所有 operational folds 的
`energy_distance`、`covariance_relative_frobenius`、`norm_energy_distance` 和
`squared_norm_energy_distance` compatibility checks 均未通过。由于存在
`LOCAL_COVARIANCE_COMPONENT_INSUFFICIENT` candidate/fold failures，按 frozen selection
rule 返回 `N3_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_INCONCLUSIVE`，`K_selected=null`。

权威 report 为
[`n3_local_shrinkage_covariance_mixture_selection_report.v1.json`](../outputs/20260908_v2/runs/n3_local_shrinkage_covariance_mixture_selection_v1_attempt_01/n3_local_shrinkage_covariance_mixture_selection_report.v1.json)，
SHA-256=`cbd16bff7f4e382f6db702099e87da4468d3213d2959492001997e4e0fb1da5a`。
该 execution 仅为 development-only selection；sealed confirmation 仍为
`RESERVED_NOT_MATERIALIZED`，未执行 final all-1000-row fit、独立 confirmation、novel OLS、
generator、evaluator、CUDA、optimizer、backward 或 training。

## N3 empirical-marginal Gaussian-copula producer contract（2026-09-08）

在 local-shrinkage-covariance family 于当前 evidence boundary `INCONCLUSIVE` 后，
`DEFINE_N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_PRODUCER_CONTRACT_V1` 定义了一个唯一候选的
non-Gaussian-marginal amplitude producer。每个 `R2_AMPLITUDE` 坐标只从 800-row fitting bank
建立 plotting-position empirical marginal；ties 使用 average rank，`Phi^{-1}` 不使用 epsilon
clamp。Gaussianized scores 使用 `n-1` 做标准化与 dependence matrix，复用既有 analytic
shrinkage，sampling 直接使用 `R_shrunk`，不做 unit-diagonal re-normalization。

权威 contract 为
[`n3_empirical_marginal_gaussian_copula_producer_contract.v1.json`](../outputs/20260908_v2/runs/n3_empirical_marginal_gaussian_copula_producer_contract_v1_attempt_01/n3_empirical_marginal_gaussian_copula_producer_contract.v1.json)，
SHA-256=`3edc9fee006f68eff6ba41205b7e4e6b6cbea3205080c4cb333689e554fba7a4`，classification 为
`DEFINE_N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_PRODUCER_CONTRACT_V1_READY`，claim 仅为
`N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_MODEL_ASSUMPTION_DEFINED`。本轮没有执行 real N3
marginal fit、Gaussianization、dependence shrinkage、sampling、model selection 或 sealed
confirmation。

## N3 empirical-marginal Gaussian-copula producer execution（2026-09-08）

`EXECUTE_N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_PRODUCER_V1` 已严格消费上述
frozen contract、同一 1000-row development pool 与五折 assignment；每 fold 只用 800 fitting
rows 建立 100 个 `R2_AMPLITUDE` empirical marginals，200 rows 仅作 held-out evaluation。
五个 fold 的 Gaussianization、analytic `R_shrunk`、正定性和单位对角 invariant 均通过；四个
DATA-vs-MODEL 与八个 MODEL-vs-MODEL bank 均 finite/nonnegative，且没有 positivity rejection。

权威 report 为
[`n3_empirical_marginal_gaussian_copula_producer_report.v1.json`](../outputs/20260908_v2/runs/n3_empirical_marginal_gaussian_copula_producer_v1_attempt_01/n3_empirical_marginal_gaussian_copula_producer_report.v1.json)，
SHA-256=`78b8e40f21015af8d704a89a83bb6b4043c0208386032b41b1ced4dd9bdeba7d`，classification 为
`N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_NOT_SUPPORTED`。只有 fold 1 和 fold 4 通过完整
compatibility；fold 0、fold 2、fold 3 未通过，因此 all-five-fold rule 不允许 development support。
描述性 decomposition 为 `MIXED`：fold 0/3 的 marginal 与 joint 均有残差，fold 2 的 joint
metrics 通过但 marginal guardrails 仍失败，fold 1/4 未见该 frozen comparison 的残余失败。
这不是对 Gaussian-copula family 的 confirmation，也不是对任何后续 intervention 的授权。

sealed confirmation 仍为 `RESERVED_NOT_MATERIALIZED` 且未消费；未执行 final all-1000 fit、独立
confirmation、novel OLS、generator、evaluator、CUDA、optimizer、backward 或 training。
`training_authorized=false`、`formal_promotion=false` 与 `generation_readiness=not_ready` 保持。
此前 local-shrinkage-covariance family 的 `INCONCLUSIVE` 结果保持原样，不能以本次相对表现替代
frozen compatibility。当前唯一后续入口是独立 planning Gate，用于决定如何归因剩余 marginal/tail
与 joint-dependence mismatch；不得自动 repair、调参、重跑或消费 sealed source。

## Frozen N3 copula generated-bank authority recovery（2026-09-08）

此前 `RESOLVE_N3_COPULA_FAILURE_MARGINAL_VS_DEPENDENCE_V1` 因历史 execution report 只有每个
bank 的 q hash、没有可复用 tensor 而停止。按独立 recovery authority，先搜索了注册的
2026-09-07/08 N3 output roots；发现的旧 r=99 producer/validation tensors 与本次 report identity
不匹配，未消费。随后严格按原 frozen contract、800-row fit、tie/Gaussianization/shrinkage、
CPU float64 RNG namespace、seed/substream、200-row bank 和原 sampling order 做了一次、且仅一次
deterministic rematerialization；没有重算 scientific metrics。

Recovery manifest 为
[`n3_frozen_copula_generated_banks_manifest.v1.json`](../outputs/20260908_v2/runs/n3_frozen_copula_bank_recovery_v1_attempt_01/n3_frozen_copula_generated_banks_manifest.v1.json)，
SHA-256=`4ed6ac4839cdb075f98d07be3162ac560ea29cc260fcbd0e272a342aed04297c`；tensor artifact
[`n3_frozen_copula_generated_banks.v1.pt`](../outputs/20260908_v2/runs/n3_frozen_copula_bank_recovery_v1_attempt_01/n3_frozen_copula_generated_banks.v1.pt)，
SHA-256=`6b9da40cf868fb9033e5c45aa05778c27199867430d2ec5627b36875d284b42b`。Semantic authority
为 `FROZEN_N3_COPULA_GENERATED_BANKS_V1`，recovery mode 为
`EXACT_DETERMINISTIC_REMATERIALIZATION`；全部 60 个 historical banks 的 expected/observed
`q_sha256` 完全相等，shape/dtype 与 finite/nonnegative checks 均通过。

该 recovery 只恢复历史 numerical sample identity，不是第二次 producer experiment，也不改变
`N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_NOT_SUPPORTED`。未完成 marginal-vs-dependence
attribution；下一独立 consumer 可读取该 authority。sealed confirmation 仍为
`RESERVED_NOT_MATERIALIZED`、`consumed=false`，`training_authorized=false`、
`formal_promotion=false` 与 `generation_readiness=not_ready` 保持。

## N3 copula failure attribution contract definition（2026-09-08）

此前 `RESOLVE_N3_COPULA_FAILURE_MARGINAL_VS_DEPENDENCE_V1` 因不存在 frozen attribution
contract/spec/consumer 而正确返回 `BLOCKED`。随后
`DEFINE_N3_COPULA_FAILURE_ATTRIBUTION_CONTRACT_V1` 已完成一次有界的 definition consumer，
只校验既有 producer contract/report、fold assignment、recovered-bank manifest 与 artifact hash，
未加载恢复的 q tensor，未计算 attribution metrics，未执行 producer fit/sampling、family selection
或 sealed confirmation。新的 write-once machine contract 为
[`n3_copula_failure_attribution_contract.v1.json`](../outputs/20260908_v2/runs/n3_copula_failure_attribution_contract_v1_attempt_01/n3_copula_failure_attribution_contract.v1.json)，
SHA-256=`333b51abe2a6b1f3e96558273d4b6037267aa80678199d3bf6d0c300bb3508df`，classification 为
`DEFINE_N3_COPULA_FAILURE_ATTRIBUTION_CONTRACT_V1_READY`，claim 为
`N3_COPULA_FAILURE_ATTRIBUTION_CONTRACT_DEFINED`。

该 contract 冻结：`M` 只来自原 report 的四个 aggregate marginal compatibility fields；每个 bank
独立做 average-rank、`u=(rank-0.5)/n`、`Phi^-1(u)`；primary `D` 为 self-rank Gaussianized
multivariate energy distance，并以同 fold 的四个 model-model 值按 linear q95 作 reference，四个
data-model comparison 采用 any-fail；Spearman、eigen/spectral 与 support/tail 仅作 descriptive
diagnostics。fold label 和五折 conservative aggregate vocabulary 已固定，且不包含
`DATA_INSTABILITY_DOMINANT`。

这只是 development-only planning contract，不产生 scientific attribution result，不选择 producer
family，不授权 `RESOLVE_N3_COPULA_FAILURE_MARGINAL_VS_DEPENDENCE_V1` 自动执行，也不授权
confirmation、novel OLS、generation 或 training；sealed confirmation 仍为
`RESERVED_NOT_MATERIALIZED`、`consumed=false`，全局 flags 保持 false/not-ready。

## N3 copula failure attribution contract V1 preflight blocked and V2 completeness revision（2026-09-09）

对 V1 contract 启动的 `EXECUTE_N3_COPULA_FAILURE_ATTRIBUTION_V1` 在读取 recovered q tensor
或计算任何 attribution metric 之前即返回 `BLOCKED`。原因是 V1 的 preflight 发现三处尚未
达到可执行定义的分支：`primary_dependence_observer.negative_or_nonfinite_policy` 没有为
`value=2*C-L-R` 的 materially-negative 与 non-finite 情形给出精确阈值/处理；Spearman
diagnostic 没有固定四个 DATA-vs-MODEL model-bank 的 per-bank 与 aggregate `S_MODEL` 语义；
spectral diagnostic 没有固定 eigenvalue median convention 或 per-bank/aggregate model-bank
输出语义。该结果是 contract/harness completeness block，不是 scientific attribution failure。

随后只执行了 `DEFINE_N3_COPULA_FAILURE_ATTRIBUTION_CONTRACT_V2`，以 V1 machine contract
[`n3_copula_failure_attribution_contract.v2.json`](../outputs/20260908_v2/runs/n3_copula_failure_attribution_contract_v2_attempt_01/n3_copula_failure_attribution_contract.v2.json)
为 base，base SHA-256=`333b51abe2a6b1f3e96558273d4b6037267aa80678199d3bf6d0c300bb3508df`，
V2 SHA-256=`5c85c4e949e5850de337d776f6782bdc791493b333ab9dd7982269e9f214eb95`，classification 为
`DEFINE_N3_COPULA_FAILURE_ATTRIBUTION_CONTRACT_V2_READY`，claim 为
`N3_COPULA_FAILURE_ATTRIBUTION_CONTRACT_V2_DEFINED`。V2 是 write-once、delta-only 的完整性修订，
不改变 V1 的 `M` authority、primary ED/q95/any-fail、fold/aggregate、bank roles 或 claim
boundary；semantic diff validation 明确通过 primary-semantics-unchanged check。

V2 只补齐三项执行约定：negative/non-finite policy 使用预先固定的 CPU float64
`tau=1e-12*max(1,2*abs(C)+abs(L)+abs(R))`，区分 numerical invalid、近零 canonicalization
与实质负值 block；Spearman 按确定顺序发出 4 个 DATA-vs-MODEL 与 8 个 MODEL-vs-MODEL
bank，`S_MODEL` 仅为四个 DATA-vs-MODEL model matrices 的等权逐元素均值；spectral 以降序
100 个实特征值的 0-based `[49,50]` 算术均值为 median，并仅将四个 DATA-vs-MODEL model
banks 纳入 model aggregate，8 个 MODEL-vs-MODEL banks 保持 descriptive reference。

V2 definition consumer 只校验既有 JSON、manifest 与 byte hash，未加载 q tensor、未计算
attribution metrics、未执行 producer fit/sampling/selection、未消费 sealed confirmation；
V1 artifact 保持 immutable。当前科学结论仍为
`N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_NOT_SUPPORTED`，`attribution_executed=false`、
`metrics_computed=false`、`sealed_confirmation_consumed=false`、`training_authorized=false`、
`formal_promotion=false`、`generation_readiness=not_ready`。V2 不构成 attribution result、P1
授权、producer-family selection 或后续 OLS/generation/training 授权；下一步只能由独立授权的
read-only attribution execution consumer 决定是否继续。

## 本轮文档与研究导航治理（2026-09-08）

本轮先完成治理/导航精简，再修复一个不改变公共合同的 evidence I/O 防御性问题。文档/导航变更
覆盖 `AGENTS.md`、`README.md`、`docs/README.md`、`docs/CURRENT_STATUS.md`、
`docs/PROJECT_MEMORY.md`、`docs/status/current_handoff.md`、`docs/status/capacity_map_hypothesis_ledger.md`、
`docs/status/experiment_history.md`、`docs/status/research_governance_state.json`、
`docs/repository_taxonomy.md`、架构/迁移/桥接和输出操作文档，
以及线程可视化目录中的 `research-map.html`。代码变更仅为
`src/ldm_project/v2/evidence/io.py`、其回归测试和
`scripts/research/resolve_n3_score_geometry_attribution.py` 的解释文本一致性修正。
主线推进4已留下的 V2 calibration consumer
`scripts/research/calibrate_and_confirm_n3_attribution_selector_v2.py` 及其 machine evidence
本轮未修改，只作为最新状态的只读输入。
治理/审查部分未修改恢复 outputs、`_wsl_restore_metadata_v4`、外部 tar/index/manifest、既有
evidence JSON、N3 阈值/selector 机制、训练代码、CLI 或公共函数签名；上方 reservation
manifest 是独立当前 run 的唯一新增 machine artifact。

本次根合同收缩只移出已有操作/架构文档中的重复步骤，保留执行、失败语义、运行时和 outputs
不可变边界；`finite_crop`/`transparent_identity` 语义归档于 v2 架构文档，正式光场 provenance
字段归档于 output artifact notes。没有删除历史实验、checkpoint、evidence 或恢复元数据。

只读验收：`git diff --check` 通过；restore state 为 `163/163 complete`，声明源文件
`33117`、源字节 `240556414616`、plan/path mapping `46/46`；HTML JavaScript 语法检查通过，
并通过临时 localhost 预览验证了研究卡片、探索/自测切换、7 题反馈、重新开始和键盘焦点。
evidence I/O 定向回归为 `23 passed`，Ruff 定向检查通过；默认全套 pytest 已执行，结果为
`1609 passed, 12 deselected, 25 subtests passed, 5 failures`。失败项属于既有主线绝对导入布局、
旧分支预期、历史 controller authority 预期和长临时路径环境问题，未改变本次 I/O 定向结果；
本轮未运行 CUDA/训练。
原始 `file://` 标签仍受浏览器安全策略拦截；窄屏/具体 viewport 的布局尚未验证，不得将该部分声明为已验证。

## 恢复后 outputs 语义合同修订（2026-09-07）

本次只修订 outputs 的治理与说明文档，不修改任何恢复的 checkpoint、图像、JSON、
日志、索引或 `_wsl_restore_metadata_v4`。只读核验确认 restore plan 的
163/163 个 archive 状态为 `complete`，声明的源文件数为 `33117`、源字节数为
`240556414616`、path mapping 为 `46`。恢复历史内容到 `20260829_v2` 为止；
`20260830_v2` 及之后的日期目录是当前 Windows-native 运行空间，最终归属以
restore plan 而不是目录修改时间推断。

`outputs/version_index.json` 和 `.md` 保留原 WSL `/home/...` 路径，只能作为历史
provenance，不能作为当前 Windows 路径解析器或 formal acceptance 输入。修改文件为
`AGENTS.md`、`README.md`、`docs/repository_taxonomy.md`、
`docs/operations/output_artifact_notes.md`、本文件和迁移记录；
`git diff --check` 通过，未运行 CUDA 或完整 pytest。本条只记录 outputs 治理边界；当前
科学下一步以本文件顶部的决策摘要和最新 N3 条目为准，不进行输出清理或自动重解析。

## N3 research history navigation

N3 OLS-native 的日期化事实、数值和被取代的下一步统一维护在
[实验历史](status/experiment_history.md)；本文件只维护上方当前决策摘要、治理边界和验证状态。
历史条目中的“下一步”只描述当时的状态，不能覆盖本摘要，也不能授予当前执行权限。
旧入口中的实验身份已逐项映射到实验历史、迁移记录、假设 ledger 或 archive；其中
`H-LOMA-GEN-PCR-PSD` 已补回实验历史，`H-LOMA-CBOGT-FIBER-MMF-M6` 保留在迁移记录。
当前入口省略的是重复正文、非权威 attempt 的逐项路径清单和过时下一步，不代表 outputs 或 machine
evidence 被删除。

## FINAL_MAINLINE_CLOSEOUT：H-LOMA-CBOGT Scaled-Chen reconstruction-repair mainline（2026-09-03）

- Current phase: `MAINLINE CLOSEOUT`
- Scientific diagnosis: `closed`
- Repair: `confirmed`
- Integration: `qualified`
- F32: `qualified`
- Visualization: `completed`
- Runtime performance: `characterized`
- `mainline_state=closed_at_current_evidence_boundary`
- Current mainline has no next scientific experiment. The future independent track is
  `REAL-HARDWARE DEPLOYMENT`; if explicitly reopened, use the task name
  `REAL_CMOS_RASPBERRY_PI_DEPLOYMENT`.

### Frozen authority chain

`RMAL → H-LOMA-CBOGT-S7-MPCDAR → H-LOMA-CBOGT-S8-MPCDAR-R16-F64-IQ →
H-LOMA-CBOGT-S9-MPCDAR-R16-F32-NQ → H-LOMA-CBOGT-S9V-FULL-PATH-VIS →
H-LOMA-CBOGT-S10-MPCDAR-R16-RUNTIME-PQ`.

Each consumer uses only its named immutable parent. Write-once non-authoritative attempts remain
for audit history and are not consumed; no later attempt is replaced by an earlier attempt.

### Frozen scientific boundary

- Within the existing optical-path/CMOS evidence boundary, task-relevant information survives
  through CMOS.
- The reconstruction failure is primarily supported as mapping/readout misalignment.
- Simple locality and raw Euclidean W-SVD are inadequate for this task; useful mean/DC material
  is present, and centered detector information is better represented by a fit-derived data-aware
  basis.
- Registered MPCDAR rank 16 is the first confirmed candidate in its preregistered powers-of-two
  family. It is
  not a minimum, intrinsic, or true rank-16 claim.

### Frozen readout and architecture

The frozen candidate is `centered=z-mu_fit; h=centered@T16; prediction=c_mean+h@V16.T`,
with `mu_fit[256]`, `c_mean[40000]`, `T16[256,16]`, and `V16[40000,16]`. The S9 F32 artifact
contains `684352` scalars, `2737408` raw tensor bytes, and `2739448` serialized bytes.
The allowed wording is: `optical front end → CMOS intensity sensing → lightweight frozen
electronic rank-16 readout`; this is hybrid optical-electronic reconstruction, and MPCDAR is
not a native CMOS optical image.

### Governance boundary

`reconstruction_manifold_supported=false`; `generative_manifold_supported=false`;
`generation_readiness=not_ready`; `training_authorized=false`; `formal_promotion=false`;
`closed_task_relevant_bottleneck_not_supported`. The hypothesis/experiment thread is
`closed / resolved to current evidence boundary`; historical hypotheses and write-once records
are retained and are not deleted.

- Not established: native optical target-like CMOS reconstruction, real physical CMOS validation,
  Raspberry Pi benchmarking, sim-to-real detector alignment, end-to-end hardware latency,
  GPU runtime, full-W runtime, quantization, generative manifold, or generation readiness.
- Future tracks remain independent: real CMOS/Raspberry Pi deployment, sim-to-real detector
  alignment, optional optimization only if a separately justified need appears, native optical
  redesign only under new explicit authorization, and generative-manifold/generation evidence.

## 历史证据导航

Windows-native migration、恢复 outputs、旧 harness、Scaled-Chen/I0/T1 以及 reconstruction closeout 的
日期化事实统一见 [实验历史](status/experiment_history.md)；迁移/桥接细节见
[Windows-native migration](migration_windows_native_20260830.md) 与
[ChatGPT ↔ Codex workflow](chatgpt_codex_workflow.md)。本文件不再复制历史 SHA 表、旧下一步或
逐次实验正文；当前允许动作只看本文件顶部摘要和 `status/current_handoff.md`。

</details>
