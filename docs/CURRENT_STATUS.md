# 当前项目状态

更新时间：2026-09-22（step9000 hybrid batched-VJP 续训预检阻断，未启动长训练）

## 当前活动执行（2026-09-22；step9000 hybrid batched-VJP 性能预检）

当前活动是 `AUTHORIZED_EXECUTE_JOINT_STRUCTURAL_SEMANTIC_CONTINUATION_V1` 的
step9000 hybrid batched-VJP 预检；目标输出目录为
`outputs/20260922_joint_structural_semantic_continuation_batched_vjp_v1/runs/attempt_01/`。
本节是本文件唯一的实时活动摘要；后续日期化 `current block` 均为历史快照，不能覆盖本节。

- accepted step9000 起点继续使用既有 structural checkpoint；旧 `attempt_03` 的 step9046 recovery 仅保留 provenance，不用于新训练。
- 新运行目录尚未创建，没有 `latest_status.json`、scientific checkpoint、`report.json` 或 `COMPLETED`。
- 当前实现固定为 `HYBRID_SEMANTIC_BATCHED_VJP`：`L_S/L_I/L_C` 使用三路 batched-VJP，结构/重建/光学梯度保持 scalar VJP；不允许静默回退。
- tuple-based batched-VJP 的固定 1 warmup + 5 次成对 benchmark：scalar median=`0.43308600000091 s`，batched median=`0.45924589999959 s`，speedup=`-6.04%`；预注册门槛为至少 `+10%`，因此性能 preflight failed，长训练未启动。
- 之前的 stack-based batched-VJP 曾约慢 `78%`；20-row direction sanity 曾在 6GB GPU 上 OOM；这些工程问题已修正为 tuple 输出和固定 10-row sanity，但当前 +10% 性能门槛仍未满足。
- accepted step9000 public DEV/S1 JSON 仅作 baseline reuse；不重复约一小时完整诊断，也不把 preflight failure 写成科学失败。
- 当前没有活动训练进程，也没有活动 heartbeat；Sources bridge 仍为 `invalid`，待工作树 clean 后重新生成。
- 当前下一动作限定为：在同一授权和同一科学合同内修复/重测性能 preflight；不得降低门槛、改损失、改 Adam、改采样或启动长训练。

## 历史 current block（2026-09-20；Three-evaluator robust semantic continuation）

按用户明确授权 `AUTHORIZED_EXECUTE_THREE_EVALUATOR_ROBUST_SEMANTIC_CONTINUATION_V1`，消费 accepted step3000 Generator、SIC attribution、冻结 E_D 独立确认 evaluator 与冻结 S1 sampler，执行唯一 Outer Gate=`THREE_EVALUATOR_ROBUST_SEMANTIC_CONTINUATION_V1_RESOLVED`。结果=`VALID_COMPLETED`，`outer_gate_passed=true`，classification=`THREE_EVALUATOR_ROBUST_SEMANTIC_CONTINUATION_SUPPORTED`，mechanism=`SEMANTIC_REALIZATION_GAIN_TRANSFERS_TO_REPAIRED_SAMPLED_MANIFOLD`。

`PROJECT MAINLINE=M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION`；`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。

- [report](../outputs/20260920_three_evaluator_robust_semantic_continuation_v1/runs/attempt_02/report.json)，SHA-256=`1d195aad5307108509349c451aa48f7015b8e27a00bfb66c8e42a56d4c4a0f92`；[step6000 checkpoint](../outputs/20260920_three_evaluator_robust_semantic_continuation_v1/runs/attempt_02/scientific_step_006000.pt)，SHA-256=`3387e104f7d33e19051a23444b0dabf1eb387c5faca3ce7150bbd1a2212c9986`。
- Stage 1 精确执行 step3000→3300、`300` updates 并通过全部预注册条件：E_D macro recall delta=`+0.0207770`，paired bootstrap 95% CI=`[+0.0145900,+0.0270100]`；`L_S/L_I/L_C` 均下降；reconstruction delta=`-0.0006815`（容差=`+0.005`）；finite/nonconstant/zero-image/power checks 全通过。随后按条件执行 Stage 2 step3300→6000、额外 `2700` updates；未超过 step6000。
- step6000 DEV macro recall：E_S/E_I/E_C/E_D=`0.1908/0.5098/0.7332/0.5678`；reconstruction=`0.0576374`，相对 step3000 delta=`-0.0036397`；光学输出 finite、nonconstant、zero-image=`0`。冻结 S1 balanced bank 的长程描述性 transfer：requested-class E_D macro recall=`0.6250`，`R_raw=1.6869`、`R_feature=0.2968`；latent S1 support authority 保持 `0.962`（CI lower=`0.953975`）。
- 仅 `generator.phase` 更新；Encoder/OLS/W/T/scale/S1/P2/Q2/E_S/E_I/E_C/E_D 冻结；Adam state 从 step3000 精确恢复，E_D 未进入 backward、optimizer 或 checkpoint selection。TRAIN/DEV 角色保持不变；protected/test/sealed reads=`0/0/0`。
- attempt-01 为修复前的 `INVALID_HARNESS` provenance，未产生 scientific report；attempt-02 为唯一有效执行结果，均保留。该结果是冻结训练与长程缩放证据，不等同于最终 generation quality、human perceptual equivalence、hardware readiness、architecture acceptance 或 promotion；`E_D` step6000 仅为 long-horizon descriptive transfer，promotion 前仍需新的确认 authority。
- [executor](../scripts/research/execute_lightgen_2d_three_evaluator_robust_semantic_continuation_v1.py) 与 focused test 已通过 compile、focused pytest（`4 passed`）；未提交 commit，既有 dirty worktree 保留。

## 历史 current block（2026-09-20；SIC independent confirmation evaluator）

按用户明确授权 `AUTHORIZED_ESTABLISH_SIC_INDEPENDENT_CONFIRMATION_EVALUATOR_V1`，消费已接受的
`BLOCKED_NEW_INDEPENDENT_CONFIRMATION_AUTHORITY_UNRESOLVED` 定义结果，执行唯一 Outer Gate=`SIC_INDEPENDENT_CONFIRMATION_EVALUATOR_ESTABLISHED`。
结果=`VALID_COMPLETED`，`outer_gate_passed=true`，classification=`SIC_INDEPENDENT_CONFIRMATION_EVALUATOR_ESTABLISHED`。

`PROJECT MAINLINE=M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION`；`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`；推荐但未授权的下一动作=`AUTHORIZED_EXECUTE_THREE_EVALUATOR_ROBUST_SEMANTIC_CONTINUATION_V1`。

- [report](../outputs/20260920_sic_independent_confirmation_evaluator_v1/runs/attempt_01/report.json)，SHA-256=`01a875ba775accc5903b58428c1a1e9468e186276ea21ffadf521dc5d8ce5db8`；[frozen E_D candidate](../outputs/20260920_sic_independent_confirmation_evaluator_v1/runs/attempt_01/final_epoch_candidate.pt)，SHA-256=`b8ece3e3bacedd5b5bec63073bee02ce7489161030b835740c681a1ebd3f9ff9`。
- 新端点 semantic authority=`SIC_INDEPENDENT_CONFIRMATION_EVALUATOR_D_V1`；架构=`PixelMLPConfirmationEvaluatorV1`，`Flatten→Linear(784,512)→ReLU→Linear(512,256)→ReLU→Linear(256,10)`。架构审计通过：E_S/E_I/E_C/Shape 均为已消费的 convolutional family；Shape 既有 qualification=`0.9234` 且已 screened out。
- canonical TRAIN-only：FIT=`49,500`、QUALIFICATION=`5,500`、split seed=`2026092003`；training seed=`2026092004`、Adam、20 epochs、final epoch only。preflight=`READY`，三步真实更新、split 对齐/无泄漏和 cyclic-label negative control 均通过。
- qualification 一次完成：accuracy=`0.9810909`、macro recall=`0.9810909`、minimum class recall=`0.9636364`，满足本任务冻结的 `0.97/0.97/0.94` 规则；E_D 在本任务中未参与 E_S/E_I/E_C/SI/SIC attribution、sampler selection、Generator 或 optical hyperparameter selection。
- data firewall：generated CMOS/optical outputs/P2Q2/S1/official test/protected/sealed reads=`0/0/0/0/0/0/0`；Generator 与 step3000 continuation trajectory 未消费。该结果只建立确认端点，不证明 SIC 训练、生成质量、硬件 readiness 或 promotion。

## 历史 current block（2026-09-19；Cross-evaluator semantic divergence attribution）

按用户明确授权 `AUTHORIZED_ATTRIBUTE_CROSS_EVALUATOR_SEMANTIC_DIVERGENCE_V1`，消费 accepted clean 2L step-0、Smooth-guided step-300、冻结 Smooth/independent evaluator、精确 TRAIN OLS/target-image identity 与 canonical-real-image positive controls，执行唯一 Outer Gate=`CROSS_EVALUATOR_SEMANTIC_DIVERGENCE_ATTRIBUTED`。结果=`VALID_COMPLETED`，`outer_gate_passed=true`，classification=`SHARED_CROSS_EVALUATOR_LOCAL_SEMANTIC_DIRECTION_SUPPORTED`。

`PROJECT MAINLINE=M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION`；`INPUT_AUTHORITY=CROSS_EVALUATOR_SEMANTIC_DIVERGENCE_ATTRIBUTED`；`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。

- [report](../outputs/20260919_cross_evaluator_semantic_divergence_v1/runs/attempt_03/report.json)，SHA-256=`c4c390df298ba82563505bec5988b0383709b57e342dfdd623efef9e70425951`；[gradient vectors](../outputs/20260919_cross_evaluator_semantic_divergence_v1/runs/attempt_03/gradient_vectors.pt)，SHA-256=`75dbe22a1af37e9caa7d6c6e414ff13cf9f66eaa1bedd97953f42b3d5a0107d2`；[directional probes](../outputs/20260919_cross_evaluator_semantic_divergence_v1/runs/attempt_03/directional_probes.json)，SHA-256=`89597dddfa54a0a58c62c5b5cdca2a73daf8f9f04f6fe67a79d93f5ef4e914bf`；[interpolation](../outputs/20260919_cross_evaluator_semantic_divergence_v1/runs/attempt_03/interpolation.json)，SHA-256=`b26d166e90332658f5fbe5b118fc94b290e76c54b2b52d329d2d93f2d925b1df`。
- Smooth canonical-real positive control=`strong`、macro recall=`0.988`；独立 FrozenMNISTEvaluator canonical validation accuracy=`0.98939997`。两者均在本任务中仅作 `DEVELOPMENT_DIAGNOSTIC`，未进入 backward、optimizer 或 evaluator search。
- TRAIN-only deterministic A/B bank：A/B 各 `20` 行、class-balanced、disjoint、`BANK_SEED=2026091901`；A 用于构造 `h_S/h_I/g_S/g_I`，B 用于固定 `epsilon={1e-4,3e-4,1e-3}` 对称非持久探针。`DEV/protected/test/sealed=0/0/0/0`，TRAIN reads=`810`。
- output-space cosine：step-0 `cos(h_S,h_I)=0.1448393`，step-300 `0.1523234`；parameter-space cosine：step-0 `cos(g_S,g_I)=0.3132313`，step-300 `0.0954316`。step-0/300 joint bisector 均 defined，且同一 A-derived `d_joint` 在 B 上对 `L_S` 与 `L_I` 都呈稳定下降符号；joint probes reconstruction/optical validity 均通过。step-300 的 `d_S/d_I` 单 evaluator 方向出现交叉效应，但不否定 joint canary。
- 固定插值 lambda=`0.25/0.50/0.75/1.00` 的 output-space gradient cosine：step-0=`0.05414/0.10162/0.15917/0.13848`，step-300=`0.07140/0.10394/0.15758/0.14626`；没有支持“越接近 CMOS 越出现单一 off-manifold evaluator shortcut”的判定。
- no-update integrity：optimizer=`false`、parameter updates=`0`、persistent probe changes=`false`、每个 probe 后 checkpoint restore=`true`、最终 runtime restore=`true`。结论只支持当前冻结 clean 2L 与 deterministic TRAIN A/B bank 上存在局部 cross-evaluator shared semantic descent direction；不支持 evaluator-robust global training success、generation acceptance、架构最优、human perceptual quality、physical hardware validity 或 promotion。推荐 successor=`DEFINE_EVALUATOR_ROBUST_SEMANTIC_TRAINING_V1`，但本轮不自动执行。
- attempt-01 为 authority/run-identity 边界的 `INVALID_HARNESS` 工程 provenance，attempt-02 为同一诊断的有效早期结果；attempt-03 在补齐 evaluator-specific preprocessing 与真实 optical-validity 归约后为当前唯一消费结果。executor 与 [focused test](../tests/v2/test_lightgen_2d_attribute_cross_evaluator_semantic_divergence_v1.py) 已通过 compile、Ruff、pytest（`5 passed`）。

## 历史 current block（2026-09-19；Canonical Smooth alignment with independent confirmation）

按用户明确授权 `AUTHORIZED_TRAIN_CANONICAL_SEMANTIC_ALIGNMENT_WITH_INDEPENDENT_CONFIRMATION_V1`，消费 accepted clean conflict-aware 2L Generator step-0、canonical attribution、Smooth contract 与一个预注册独立 canonical evaluator，执行唯一 Outer Gate=`CANONICAL_SEMANTIC_ALIGNMENT_WITH_INDEPENDENT_CONFIRMATION_V1_RESOLVED`。结果=`VALID_COMPLETED`，`outer_gate_passed=false`，classification=`SMOOTH_GUIDED_ALIGNMENT_WITHOUT_INDEPENDENT_CONFIRMATION`（Case B）。

`PROJECT MAINLINE=M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION`；`INPUT_AUTHORITY=CANONICAL_SEMANTIC_REALIZATION_FAILURE_ATTRIBUTED`；`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。

- [report](../outputs/20260919_canonical_semantic_alignment_with_independent_confirmation_v1/runs/attempt_01/report.json)，SHA-256=`440230967def4f758a9d72a7999aac34d6127bce487fb078a2320a9b02ab4423`；[preflight](../outputs/20260919_canonical_semantic_alignment_with_independent_confirmation_v1/runs/attempt_01/preflight.json)，SHA-256=`d38ec15f75dc1d4cf1a9fc9433c934312068843f2be4a6d39970443d3d307d7c`；[step-300 checkpoint](../outputs/20260919_canonical_semantic_alignment_with_independent_confirmation_v1/runs/attempt_01/scientific_step_000300.pt)，SHA-256=`5efdcfeef9c10a403223c43ae76c464591279261fc593e8e3c1c42016efc1843`。
- 独立 confirmation authority 唯一解析为历史 accepted `FrozenMNISTEvaluator v1` checkpoint SHA=`16a42fad849ed9af249fa65bed9567a8fabc011fcc9d107061f69cdb27f21684`，canonical real-image validation accuracy=`0.98939997`；当前 teacher checkpoint 因已参与既有 teacher objective 排除，Smooth 因本轮训练/development oracle 排除，Shape evaluator 因 validation accuracy=`0.9234` 且已被 ledger 明确排除。该 evaluator 只在 DEV step `0/300` 做 confirmation，未进入 backward 或 probe。
- 仅 `generator.phase` 更新；Encoder/OLS/scale/`T=I_100`/`W_OLS`/latent normalization/P2/Q2/Smooth/independent confirmation 全部冻结。canonical Smooth feature loss=`1.0439515→0.9424558`（delta=`-0.1014957`），Smooth DEV macro recall=`0.1030→0.1088`（delta=`+0.0058`），paired feature distance delta=`-0.0513843`；centroid matched-nearest fraction 未移动（delta=`0`）。
- 独立 confirmation macro recall=`0.1698→0.1456`（delta=`-0.0242`），paired bootstrap `B=999`、95% CI=`[-0.0322,-0.01599]`，未支持 independent semantic transfer。DEV reconstruction=`0.0687409→0.0671059`，delta=`-0.0016350`，`+0.005` constraint 通过；raw CMOS finite/nonconstant=`1.0/1.0`、zero-image=`0`。
- P2+Q2 secondary transfer 已生成；final `R_raw=2.5831490`、`R_feature=8.2150978`。该 transfer 不是 acceptance gate。data firewall：TRAIN=`3000`、DEV=`20000`，preflight TRAIN=`30`，protected/test/sealed=`0/0/0`；CUDA preflight=`READY`，checkpoints=`0/100/200/300`。
- 结论只支持：直接 canonical Smooth target-image alignment 可在当前 clean 2L Generator 上改善 Smooth development objective/局部 feature distance，但未转移到预注册独立 evaluator；不支持 canonical semantic recovery、generation success、teacher/architecture optimality、physical hardware validity 或 promotion。不得自动继续训练、调参、替换 evaluator、改架构或 promotion。
- [executor](../scripts/research/execute_lightgen_2d_train_canonical_semantic_alignment_with_independent_confirmation_v1.py) 与 [focused test](../tests/v2/test_lightgen_2d_train_canonical_semantic_alignment_with_independent_confirmation_v1.py) 已通过 compile、Ruff、pytest（`5 passed`）。

## 历史 current block（2026-09-19；Canonical semantic realization failure attribution）

按用户明确授权 `AUTHORIZED_ATTRIBUTE_CANONICAL_SEMANTIC_REALIZATION_FAILURE_V1`，消费 accepted 的 target-image semantic-alignment result、clean step-0 与 failed step-300 Generator checkpoint、FrozenMNISTEvaluator、SmoothMNISTEvaluator 与 TRAIN OLS/sample/target identity，执行唯一 Outer Gate=`CANONICAL_SEMANTIC_REALIZATION_FAILURE_ATTRIBUTED`。结果=`VALID_COMPLETED`，classification=`TEACHER_TO_CANONICAL_OBJECTIVE_TRANSFER_MISMATCH_SUPPORTED`。

`PROJECT MAINLINE=M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION`；`INPUT_AUTHORITY=CANONICAL_SEMANTIC_REALIZATION_FAILURE_ATTRIBUTED`；`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。

- [report](../outputs/20260919_canonical_semantic_realization_failure_attribution_v1/runs/attempt_03/report.json)，SHA-256=`0efe156ac0bbdf6e7a7d1cfe40f6fd98483fa855f4e7746570f4511fef7668e2`；[gradient vectors](../outputs/20260919_canonical_semantic_realization_failure_attribution_v1/runs/attempt_03/gradient_vectors.pt)，SHA-256=`c3f1e0f0f4e0abed71946cbc025c254387e9987bc8a6f4301aea4e3c39bf7c43`；[directional probes](../outputs/20260919_canonical_semantic_realization_failure_attribution_v1/runs/attempt_03/directional_probes.json)，SHA-256=`6271db595c55bebeaebb5f411824b866318e393bd6851a73823d08647068738d`。
- 固定 TRAIN-only class-balanced bank：A/B 各 `20` 行，`BANK_SEED=2026091901`，A/B disjoint；没有索引 DEV。step-0/300 均完成 `h_T/h_S` output-space、`g_T/g_S/g_total` parameter-space attribution；`cos(h_T,h_S)=+0.07631/+0.07338`，`cos(g_T,g_S)=+0.15877/+0.10191`，`cos(g_total,g_S)=+0.29854/+0.30322`。
- B 对称方向结果：`d_T` 在 step-0 与 step-300 均使 teacher `L_T` 下降，但 Smooth `L_S` 上升；`d_S` 在两个 checkpoint 均使 Smooth `L_S` 下降。该交叉样本模式支持当前 teacher objective 与 canonical Smooth semantic direction mismatch；不支持 clean 2L 缺少可迁移 canonical local direction。
- Smooth 使用的角色更新为 `SMOOTH_ROLE_AFTER_ATTRIBUTION=DEVELOPMENT_DIAGNOSTIC`；这不使此前 Smooth measurements 失效。推荐但不自动执行的 successor=`DEFINE_CANONICAL_ALIGNMENT_TRAINING_WITH_INDEPENDENT_CONFIRMATION_V1`，必须先注册 distinct independent confirmation evaluator。
- no-update integrity：optimizer=`false`、parameter updates=`0`、persistent probe changes=`false`、每个 probe 后 checkpoint restore=`true`、最终 runtime restore=`true`；preflight=`READY`。data reads：TRAIN=`800`（重复 probe reads），DEV/protected/test/sealed=`0/0/0/0`。
- attempt-01 在最终 runtime 全量 state 比较处 `INVALID_HARNESS` 停止，attempt-02 在 artifact 序列化前 `INVALID_HARNESS` 停止；二者均没有 scientific report/classification，保留为工程 provenance。有效结果只消费 attempt-03；不自动训练、调参、续训、改架构、替换 evaluator 或 promotion。
- [executor](../scripts/research/execute_lightgen_2d_attribute_canonical_semantic_realization_failure_v1.py) 与 [focused test](../tests/v2/test_lightgen_2d_attribute_canonical_semantic_realization_failure_v1.py) 已通过 compile、Ruff、pytest（`3 passed`）。

## 历史 current block（2026-09-19；Generator target-image semantic alignment）

按用户明确授权 `AUTHORIZED_TRAIN_GENERATOR_TARGET_IMAGE_SEMANTIC_ALIGNMENT_V1`，从 accepted clean conflict-aware 2L Generator step-0 执行唯一 Outer Gate=`GENERATOR_TARGET_IMAGE_SEMANTIC_ALIGNMENT_V1_RESOLVED` 的 300-update Generator-only arm。结果=`VALID_COMPLETED`，`outer_gate_passed=false`，classification=`TARGET_FEATURE_ALIGNMENT_WITHOUT_INDEPENDENT_SEMANTIC_TRANSFER`（Case B）。

`PROJECT MAINLINE=M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION`；`INPUT_AUTHORITY=GENERATOR_SEMANTIC_CONTROLLABILITY_ATTRIBUTION_RESOLVED`；`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。

- [report](../outputs/20260919_generator_target_image_semantic_alignment_v1/runs/attempt_01/report.json)，SHA-256=`500289c9e17e28500e5b103a5d9bfcd7b92b1ace5bd16fdccd9dea683e570f68`；[preflight](../outputs/20260919_generator_target_image_semantic_alignment_v1/runs/attempt_01/preflight.json)，SHA-256=`a379e1679263b6be41b8adc3e02ce5a1bb30cf3e7a48427c3293edf40f073bfc`；[step-300 checkpoint](../outputs/20260919_generator_target_image_semantic_alignment_v1/runs/attempt_01/scientific_step_000300.pt)，SHA-256=`7b51f01e0965295f059e0b7f52de1b7c68cdb5cb9ec438c4bca69d4052adcacf`。
- TRAIN teacher=`FrozenMNISTEvaluator v1`，固定 128D pre-classifier embedding；canonical DEV positive control macro recall=`0.9876`；一次 TRAIN calibration 将 `delta_align=0.0245538596` 的 weighted alignment gradient 固定到历史 authority `0.0019317994`。仅 `generator.phase` 更新，Encoder/OLS/W/P2/Q2/evaluators 全部冻结；旧 TN、label CE、scalar weighted-sum backward 均未进入。
- fixed TRAIN alignment diagnostic `L_align=` `1.078848→1.046632`（0→300）；独立 Smooth DEV macro recall=`0.1030→0.1030`，delta=`0.0000`，paired 95% CI=`[-0.0008,+0.0008]`，因此没有 independent semantic transfer。Smooth feature paired distance=`1.0121→0.9999` 有局部下降，但 centroid matched-nearest fraction=`0.1→0.1`，10 个 CMOS centroids 仍全部最近 real class 8。
- DEV reconstruction=`0.0687409→0.0672903`，delta=`-0.0014505`，`+0.005` constraint 通过；raw CMOS finite/nonconstant=`1.0/1.0`、zero-image=`0`。固定 P2+Q2 transfer 的 `R_raw=1.5288→1.5635`、`R_feature=3.0552→3.2780`，未改善。
- data firewall：scientific TRAIN=`3000`、DEV=`20000`；protected/test/sealed=`0/0/0`；CUDA preflight=`READY`，three-step real update、negative control、Generator-only gradient firewall、optimizer/moment restoration 均通过。四个 scientific checkpoints=`0/100/200/300`，latest state=`COMPLETED`。
- 结论只支持：在 frozen current Encoder/OLS manifold 与 clean 2L Generator 下，paired target-image feature alignment 可改善 TRAIN-side teacher feature loss，但在本次 300-update scope 内未改善独立 Smooth semantic realization；不支持 generation acceptance、teacher/architecture optimality、physical hardware validity 或 promotion。下一动作保持 `NONE`，不自动 teacher search、调参、续训或改架构。
- [executor](../scripts/research/execute_lightgen_2d_generator_target_image_semantic_alignment_v1.py) 与 [focused test](../tests/v2/test_lightgen_2d_generator_target_image_semantic_alignment_v1.py) 已通过 Ruff、compile、pytest（`5 passed`）。

## Previous current block（2026-09-19；CMOS→Smooth semantic realization）

`PROJECT MAINLINE=M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION`；`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。
`INPUT_AUTHORITY=GENERATOR_SEMANTIC_CONTROLLABILITY_ATTRIBUTION_RESOLVED`。

按用户明确授权 `AUTHORIZED_RESOLVE_CMOS_TO_SMOOTH_SEMANTIC_REALIZATION_V1`，消费 accepted 的同一 5000-row DEV images/labels、raw CMOS、CMOS probe、Smooth checkpoint 与 preprocessing contract，完成唯一 Outer Gate=`CMOS_TO_SMOOTH_SEMANTIC_REALIZATION_RESOLVED`。结果=`VALID_COMPLETED`，最终 classification=`CMOS_CLASS_CODE_PRESERVED_BUT_IMAGE_SEMANTIC_REALIZATION_FAILED`。

- [final report](../outputs/20260919_cmos_to_smooth_semantic_realization_v1/runs/attempt_04/report.json)，SHA-256=`46fcfd60e5d240a92a9c4ca472f8bb88a90c641474ba11eea9db9da7236dba65`；[finalization manifest](../outputs/20260919_cmos_to_smooth_semantic_realization_v1/runs/attempt_04/source_manifest.json)，SHA-256=`76299a681ed9a710ec4431152ee1f142553f05b0126de424ee40601fb72871e8`。final report 只修正 attempt-02 的分类谓词，复用其 immutable measurement artifacts，没有新 forward。
- Smooth contract audit 通过：canonical real-image positive control macro recall/accuracy=`0.9880/0.9880`；同一 contract 在 DEV raw CMOS 上 exact reproduce Smooth macro recall=`0.1030`，class-8 recall=`0.998`；无 double/missing normalization、range/channel/pooling/class-map/order/checkpoint mismatch。
- accepted CMOS linear probe DEV macro recall=`0.9708`；同一 5000 rows 的 `probe-correct/Smooth-wrong=`4359，反向=`20`。固定 Smooth final embedding 的 L2-normalized Euclidean geometry：paired mean=`1.0121`，same-class reference=`1.0361`，different-class=`1.0655`；CMOS→real centroid matched-nearest fraction=`0.1`，10 个 CMOS centroids 全部最近 real class 8。该 centroid collapse 是 canonical image-semantic realization failure 的主要证据；不支持“CMOS 已对齐 Smooth feature、仅 decision head 失配”的 Case D。
- pixel context：accepted clean step-0 `L_rec=` `0.0687409`；fixed Smooth-view paired MSE=`0.0788718`。optional nonlinear context 未消费，因为现有 nonlinear artifact 只有 200 rows 且没有 exact 5000-row binding。
- data firewall：DEV=`5000`，为复现既有 probe 标准化仅读取 accepted TRAIN raw CMOS=`55000`；protected/test/sealed=`0/0/0`；optical forward regeneration/backward/update=`false/false/0`；evaluator/probe retraining=`false/false`。推荐的单一 successor 是 `TRAIN_GENERATOR_TARGET_IMAGE_SEMANTIC_ALIGNMENT_V1`，本轮未执行、未授权 promotion。
- attempt-01/03 仅保留为本次 harness/资源停止 provenance；attempt-02 的旧分类报告已由 attempt-04 supersede，accepted measurement tensors 未覆盖。新 executor 与 `6 passed` focused tests、compile、Ruff 均通过；`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。

## Previous current block（2026-09-19；OLS→CMOS semantic localization）

按用户明确授权 `AUTHORIZED_RESUME_LOCALIZE_OLS_TO_CMOS_SEMANTIC_INFORMATION_V1`，消费 accepted clean 2L Encoder/OLS/Generator、匹配的 DEV OLS/CMOS 与 TRAIN OLS authority；在未改变任何 optical parameter 的前提下，物化唯一 clean TRAIN raw-CMOS authority，并用固定 TRAIN-only multinomial logistic + L2 probe 完成一次 DEV measurement。唯一 Outer Gate=`OLS_TO_CMOS_SEMANTIC_INFORMATION_LOCALIZED`，结果=`VALID_COMPLETED`，classification=`SEMANTIC_INFORMATION_LOCALIZATION_MIXED`。

- [report](../outputs/20260919_ols_to_cmos_semantic_information_localization_v1/runs/attempt_01/report.json)，SHA-256=`17b7d19ef2528dd2039f94e5e97f7d5e24b54a4dd70047413e9e9daf200c7d98`；[probe metrics](../outputs/20260919_ols_to_cmos_semantic_information_localization_v1/runs/attempt_01/probe_metrics.json)，SHA-256=`0c63c0485eded59d40fd3cc6374b0912dad8786b143488f5ed1b705c37056724`；[probe weights](../outputs/20260919_ols_to_cmos_semantic_information_localization_v1/runs/attempt_01/probe_weights.pt)，SHA-256=`b76e1d22420bd48e620a3d293ca15e8d6e04ab192df5cc5bee8c1cdb5310b383`。
- 唯一 TRAIN CMOS authority 为 [train_raw_cmos.pt](../outputs/20260919_ols_to_cmos_semantic_information_localization_v1/runs/attempt_01/train_raw_cmos.pt)，shape=`[55000,1,200,200]`、dtype=`float32`、file SHA-256=`1271ba91c396f2a5d730b72f5ab425eff9127b7f987b371e0a0ab13f9f9494cd`、tensor SHA-256=`63efe4bd678440af0a01718a13d7c4dccbb574c4be9c420c84ff47ceaefe3ac8`；TRAIN indices/labels row binding 已证明，zero/constant images=`0/0`。
- 固定 probe 的 DEV macro recall：OLS=`0.9226`（TRAIN=`0.9297`），CMOS=`0.9708`（TRAIN=`0.9873`）；一次 deterministic TRAIN-label permutation 的 OLS DEV macro recall=`0.1190`。OLS→CMOS retention=`1.05224`，说明当前证据不是 OLS→Generator→CMOS 的语义衰减。
- accepted clean DEV Smooth reference=`0.1030`，且由已物化 DEV CMOS 重算 exact reproduce；matched transitions 为 OLS-correct→CMOS-wrong=`56`、OLS-wrong→CMOS-correct=`297`。Smooth 仅 class 8 recall=`0.998`，其余类别接近零，因此当前只能保守归类 mixed，不能把 probe decodability 解释为 mutual information 或 optical inference readiness。
- data firewall：TRAIN forward/materialized=`55000`，DEV measurement=`5000`，protected/test/sealed=`0/0/0`；optical optimizer/backward/update=`false/false/0`。曾发生的内存 harness 修复与 partial-task resume 均在 report provenance 中记录，没有重复 optical forward 或覆盖 write-once authority。
- [executor](../scripts/research/execute_lightgen_2d_ols_to_cmos_semantic_information_localization_v1.py) 与 [focused test](../tests/v2/test_lightgen_2d_ols_to_cmos_semantic_information_localization_v1.py) 已通过 Ruff、compile、pytest（`5 passed`）。`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`；不自动执行 coupling repair、encoder retraining、evaluator replacement、generation 或 promotion。

## Previous current block（2026-09-18；latest bounded execution before OLS→CMOS localization）

按用户明确授权 `AUTHORIZED_EXECUTE_MINIMAL_NEAR_IDENTITY_NONLINEAR_GENERATOR_V1`，从 accepted clean 2L conflict-aware Generator step-0（仅消费 accepted conflict-aware step-0，不消费失败/线性 3L endpoint）执行一次 bounded near-identity nonlinear Generator successor。唯一新增变量是在既有第一层 Generator endpoint、第二个 phase plane 之前插入一个冻结 `I_ref` 标定的 scalar `eta` 算法相位算子：`rho=|E|^2/(I_ref+1e-12)`、`N_eta(E)=E*exp(i*eta*rho)`；`eta=0` 为 identity。没有新增 propagation/FFT/crop/aperture/relay/normalization；phase、encoder、OLS、P2/Q2、evaluators、loss weights、数据顺序、Adam family 与其余 frozen context 均保持不变。该算子是 MODEL_ASSUMPTION/algorithmic surrogate，不是 Kerr、Te、fiber facet 或 hardware validation。结果=`VALID_COMPLETED`，Outer Gate=`MINIMAL_NEAR_IDENTITY_NONLINEAR_GENERATOR_V1_RESOLVED`，`outer_gate_passed=false`，classification=`NONLINEAR_LOCAL_GAIN_WITHOUT_DATASET_RECOVERY`（Case B）。

- [report](../outputs/20260918_minimal_near_identity_nonlinear_generator_v1/runs/attempt_02/report.json)，SHA-256=`d2526b49540d6ea03483f51413587b09b8195440af3c72cd1925f3673c7f5673`；[preflight](../outputs/20260918_minimal_near_identity_nonlinear_generator_v1/runs/attempt_02/preflight.json)，SHA-256=`fc1bf02eb6dd62ffd1b4d61b671172c8af85069fe456a5c78b1ceda232179b08`；[step-300 checkpoint](../outputs/20260918_minimal_near_identity_nonlinear_generator_v1/runs/attempt_02/scientific_step_000300.pt)，SHA-256=`630447422b1cac7ad900a98b25712ed53642b2c98f4ca64e7bde3617694bd32a`。
- clean 2L starting authority 为 conflict-aware report SHA=`dc05742fae2333e21a1d123fd93d4032b9e67676ae4e424ea3c68e5d6db8c904`、step-0 SHA=`a61dd84728aa220aa567ab6cd5f9daa788b21e76be3c506054c2b7c0a23dcd34`；failed/linear 3L step-300 明确未加载。固定 calibration bank=`10` TRAIN rows，`I_ref=0.9120797964019747`，insertion plane=`lightgen_generator_layer_1`，`z=0.6295 m`；[Stage 1 gradients](../outputs/20260918_minimal_near_identity_nonlinear_generator_v1/runs/attempt_02/stage1_gradient_vectors.pt)，SHA-256=`14b83d3ae8b2fc3c16ca283d789d7fbdae7ce393677a6b01b33fbf0f667d001b`；[directional probes](../outputs/20260918_minimal_near_identity_nonlinear_generator_v1/runs/attempt_02/stage1_directional_probes.json)，SHA-256=`3ebb4c42e80039602bb238b2b170b16a7216821f6dd3b95fe4cce194d5727dff`。
- eta=0 function-preserving check=`accepted_for_stage1=true`：generator input、插入后 field、两层 endpoint、final output、raw CMOS、reconstruction scalar、两套 evaluator input/logits 均 `max_abs=0`；phase shape 与 record count 未变。Stage 1 A/B eta semantic gradients=`-0.4071133/-1.1347258`，方向=`+1`，同号且 beneficial；B 对称探针 `eps={1e-4,3e-4,1e-3}` 的 Smooth slopes=`-1.4567375/-1.1968613/-1.1153221`，raw validity 全部通过。
- preflight=`READY`：3-step real update、negative control、frozen encoder/OLS/evaluator firewall、eta finite、phase moments copied、eta moments zero/shared step counter 均通过；正式 scope=`300` updates，checkpoints/DEV=`0/100/200/300`，CUDA=`true`。data reads：Stage 1 TRAIN=`10`、preflight TRAIN=`30`、scientific TRAIN=`3000`、DEV=`20000`；protected/test/sealed=`0`。
- independent Smooth DEV macro recall=`0.1030→0.1028→0.1028→0.1024`，step0→300 delta=`-0.0006`，paired 95% CI=`[-0.0020,+0.0006]`，semantic recovery gate 未支持。DEV reconstruction=`0.0687409→0.0671431`，delta=`-0.0015977`，`+0.005` 约束通过；raw validity=`finite/nonconstant 1.0/1.0`、zero-image=`0`。
- exact P2+Q2/real-control transfer：`R_feature=3.0552277→1.5064399`（局部 feature distance 改善），但 `R_raw=1.5288204→1.8741536`（raw distribution 未改善）；`R_u=1.5095471`、`R_z=1.8283132`、`R_phi=0.9937009` 保持。 [P2+Q2 generated CMOS](../outputs/20260918_minimal_near_identity_nonlinear_generator_v1/runs/attempt_02/minimal_near_identity_nonlinear_p2q2_generated_cmos.pt)，SHA-256=`7f655750c7d41df093cf250aa47a885d6214428878bd92e81fb945c9c30f4dad`；[paired bootstrap](../outputs/20260918_minimal_near_identity_nonlinear_generator_v1/runs/attempt_02/paired_bootstrap.pt)，SHA-256=`83d7bd8f5c2f9356059b0327971e6989c0b8134a029083132d058bb211d350f9`。
- attempt-01 在 Stage 1 前因现有 distribution-Q 的 logical-batch boundary 被 fail-closed（scientific updates=`0`）并保留为 harness provenance；最小修复显式关闭半批诊断中的 distribution term 后，attempt-02 完成有效执行。该修复未改变 Stage 2 objective、数据、模型或 acceptance gate；无 retry、无第二 scientific arm、无 sweep/调参/续训。
- 结论只支持：该单一 normalized intensity-dependent eta 方向在固定 TRAIN A→B 局部诊断中可控，并在当前 P2/Q2 feature transfer 上出现局部改善；不支持 dataset-level semantic recovery、global architecture inadequacy、generation success、physical hardware validity 或 promotion。`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`，不自动追加新的 nonlinear family、insert sweep、4L/ReLU/N6 或 promotion。
- [executor](../scripts/research/execute_lightgen_2d_minimal_near_identity_nonlinear_generator_v1.py) 与 [focused test](../tests/v2/test_lightgen_2d_minimal_near_identity_nonlinear_generator_v1.py) 已通过 compile/lint/pytest（`3 passed`）。

## Minimal Generator capacity expansion（2026-09-18；previous bounded execution）

按用户明确授权 `AUTHORIZED_EXECUTE_MINIMAL_GENERATOR_CAPACITY_EXPANSION_V1`，从 accepted 2L conflict-aware Generator endpoint 启动一次且仅一次 minimal 3L capacity-expansion successor。唯一结构变量是将现有第二个 `1.50 mm` propagation interval 对称拆为 `0.75 mm + 0.75 mm`，插入一个初始为零的 trainable phase plane；crop、aperture、sampling、normalization、relay、encoder、OLS、P2/Q2、evaluators、loss weights、数据顺序与 optimizer family 均冻结，旧两平面 Adam moments 复制、新平面 moments 置零。结果=`VALID_COMPLETED`，Outer Gate=`MINIMAL_GENERATOR_CAPACITY_EXPANSION_V1_RESOLVED`，`outer_gate_passed=false`，classification=`LOCAL_CONTROLLABILITY_GAIN_WITHOUT_DATASET_RECOVERY`（Case B）。

- [report](../outputs/20260918_minimal_generator_capacity_expansion_v1/runs/attempt_01/report.json)，SHA-256=`ea8fb6b3e766a667461cfa061d9d22f42306d34eb0757aeb0d0e87d6202b817f`；[preflight](../outputs/20260918_minimal_generator_capacity_expansion_v1/runs/attempt_01/preflight.json)，SHA-256=`a229fa926b52307e4768682a476c717a483f85de73e8f94a1185c43d0f7212a1`；[step-300 checkpoint](../outputs/20260918_minimal_generator_capacity_expansion_v1/runs/attempt_01/scientific_step_000300.pt)，SHA-256=`7ee9c3782b3b5ddff1ebf064eef4836d7e18a63385fff0dbf64bbb3b2cbac517`。
- function-preserving check=`accepted_for_stage1=true`，旧平面/输入/relay/crop/aperture/sampling 均保持；raw CMOS relative error=`9.2764e-7`，isolated half-split FFT control relative error=`6.6261e-7`。Stage 1 `supported=true`：new-plane Smooth gradient norm=`0.1424059`、Smooth gradient energy fraction=`0.2021425`，3L/2L fixed-scale Smooth response ratios=`1.1105/1.0947/1.1173`。
- preflight=`READY`：3-step real update、full-runtime optimizer negative control、frozen gradient firewall、new-plane zero-moment initialization 均通过；CUDA=`true`；TRAIN/DEV=`3000/20000`，protected/test/sealed=`0`。正式 scope 为 `300` updates，checkpoints/evaluations=`0/100/200/300`。
- independent Smooth DEV macro recall=`0.1030→0.1036→0.1034→0.1032`，step0→300 delta=`+0.0002`，paired 95% CI=`[-0.0004,+0.0008]`，CI lower 未大于 `0`，因此 DEV semantic recovery 未支持。DEV reconstruction=`0.0687409→0.0663280`，delta=`-0.0024129`，`+0.005` 约束通过；raw validity=`finite/nonconstant 1.0/1.0`、zero-image=`0`。
- exact P2+Q2/real-control transfer：`R_raw=1.5288204→2.0526421`、`R_feature=3.0552277→3.6763498`；`R_u=1.5095471`、`R_z=1.8283132` 保持；generated class histogram=`[2,1,44,65,1,44,3,16,17,7]`、dominant fraction=`0.325`、normalized entropy=`0.74805`。输出 [expanded P2+Q2 CMOS](../outputs/20260918_minimal_generator_capacity_expansion_v1/runs/attempt_01/expanded_generator_p2q2_generated_cmos.pt)，SHA-256=`ebe6e7b1bae3df94d474fbb4d6e4299e2bd1671e769c2d68697e5d7fdc1c968c`。
- 结论只支持：增加一个 function-preserving、可训练的 phase plane 扩大了当前 frozen Generator 的局部语义可控性；但固定 P2/Q2 transfer 的 `R_feature` 从 `3.0552` 上升至 `3.6763`，未改善。300-update dataset-level DEV semantic gate 未通过，不能宣称 semantic recovery、全局容量充分、generation success 或 promotion。`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`，不自动追加 4L/sweep、调参、续训、架构变更或 promotion。
- [executor](../scripts/research/execute_lightgen_2d_minimal_generator_capacity_expansion_v1.py) 与 [focused test](../tests/v2/test_lightgen_2d_minimal_generator_capacity_expansion_v1.py) 已通过 compile/lint/pytest（`4 passed`）；第一次同授权启动留下的未完成 preflight 目录仅作 aborted provenance，不作为科学结果消费。

## Conflict-aware Generator semantic recovery（2026-09-18；previous bounded execution）

按用户明确授权 `AUTHORIZED_EXECUTE_CONFLICT_AWARE_GENERATOR_SEMANTIC_RECOVERY_V1`，从 accepted frozen-latent semantic-recovery successor 的 step-0 状态启动一次且仅一次 conflict-aware Generator-only successor。仅改变 multi-objective gradient-composition rule：`g_total=g_sem+g_rec_safe+g_op_safe`，其中只有与 semantic descent 负内积的 weighted reconstruction/optical-power component 被以固定 `eps_proj=1e-12` 投影移除；Encoder、OLS、scale、`T=I_100`、`W_OLS`、latent normalization、P2/Q2、evaluators、architecture、loss weights、optimizer state、batch order 均冻结。结果=`VALID_COMPLETED`，Outer Gate=`CONFLICT_AWARE_GENERATOR_SEMANTIC_RECOVERY_V1_RESOLVED`，classification=`CONFLICT_AWARE_SEMANTIC_RECOVERY_NOT_SUPPORTED`（Case C）。

- [conflict-aware report](../outputs/20260918_conflict_aware_generator_semantic_recovery_v1/runs/attempt_01/report.json)，SHA-256=`dc05742fae2333e21a1d123fd93d4032b9e67676ae4e424ea3c68e5d6db8c904`；[preflight](../outputs/20260918_conflict_aware_generator_semantic_recovery_v1/runs/attempt_01/preflight.json)；[step-300 checkpoint](../outputs/20260918_conflict_aware_generator_semantic_recovery_v1/runs/attempt_01/scientific_step_000300.pt)，SHA-256=`9e561f4a95da2a6962e6e6bcf0ac3ebb0680e974a8ef85c2c99cf18425b73d35`；[P2+Q2 generated CMOS](../outputs/20260918_conflict_aware_generator_semantic_recovery_v1/runs/attempt_01/successor_p2q2_generated_cmos.pt)，SHA-256=`dfc6eb2e1744314f4518aef3b3c734c60ce80b728293741421c4d923c3fa7f44`。
- preflight=`READY`：3-step real update、full-runtime optimizer negative control、Generator-only gradient firewall、projection orthogonality、same batch/order/optimizer/`beta=2.4509953404`/`gamma=0.5330424285`/`delta=0.0065783691` 均通过；optimizer=`PRIOR_SEMANTIC_RECOVERY_STEP0_STATE_RESTORED`；old TN 与 scalar weighted-sum backward 均未进入。
- fixed TRAIN diagnostic composition trajectory：`g_sem`=`0.0019317994→0.0017961984`，`g_rec`=`0.0014377991→0.0014116786`，`g_op`=`0.0003778288→0.0003554131`；`cos(g_rec,g_sem)`=`+0.07053→+0.04057`，`cos(g_op,g_sem)`=`-0.00553→+0.04993`，`cos(g_total,g_sem)`=`+0.81197→+0.79359`。仅 step-0 的 optical-power component 被投影，projection fraction=`0.005534`；后续 rec/op projection fraction=`0`。
- independent Smooth DEV macro recall trajectory=`0.1030→0.1034→0.1038→0.1036`（0/100/200/300），step0→300 delta=`+0.0006`，paired bootstrap `B=999`、seed=`271828`、95% CI=`[0,+0.00121]`；CI lower 未大于 `0`，semantic recovery unsupported。DEV reconstruction=`0.0687409→0.0682367→0.0677573→0.0673604`，delta=`-0.0013805`，constraint 通过。
- exact P2+Q2/real-control transfer：`R_raw=1.5288204→1.5574754`，`R_feature=3.0552277→3.1940135`，R_feature 未下降；generated class histogram=`[1,3,48,68,0,44,1,18,10,7]`，dominant fraction=`0.34`，normalized entropy=`0.71321`；raw finite/nonconstant=`1.0/1.0`、zero-image=`0`。
- 因 Smooth CI lower=`0`，classification 为 Case C：conflict removal 未恢复 dataset-level semantics，增加对当前 Generator controllable-subspace/capacity limitation 的优先级，但不构成 global architecture inadequacy。TRAIN/DEV reads=`3000/20000`，protected/test/sealed=`0`；不自动追加训练、architecture change、调参、promotion 或新研究 arm。
- [executor](../scripts/research/execute_lightgen_2d_conflict_aware_generator_semantic_recovery_v1.py) 与 [focused test](../tests/v2/test_lightgen_2d_conflict_aware_generator_semantic_recovery_v1.py) 已通过 compile/lint/pytest。`PROJECT MAINLINE=M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION`；`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。

## Generator semantic-controllability attribution（2026-09-18；accepted input authority）

按用户明确授权 `AUTHORIZED_EXECUTE_FROZEN_LATENT_GENERATOR_SEMANTIC_RECOVERY_V1` 的 accepted result，执行一次且仅一次 no-update attribution consumer。结果=`VALID_COMPLETED`，Outer Gate=`GENERATOR_SEMANTIC_CONTROLLABILITY_ATTRIBUTION_RESOLVED`，primary attribution=`GENERATOR_SEMANTIC_CONTROLLABILITY_MIXED`，secondary=`RECONSTRUCTION_SEMANTIC_GRADIENT_CONFLICT_SUPPORTED`。该结果只覆盖 accepted calibration batch 与 frozen runtime，不构成 global architecture、generation acceptance、physical validity 或 promotion 结论。

- [attribution report](../outputs/20260918_generator_semantic_controllability_attribution_v1/runs/attempt_01/report.json)，SHA-256=`3304262b2cd5fbff22e423c415984c14eeb1a2906871ad45e3a62265479d89b5`；[gradient vectors](../outputs/20260918_generator_semantic_controllability_attribution_v1/runs/attempt_01/gradient_vectors.pt)，SHA-256=`ae4515640b0005b9e3c56328d022c430e97ecc3cdba0a9dcce5f6b29c7289da1`；[directional probes](../outputs/20260918_generator_semantic_controllability_attribution_v1/runs/attempt_01/directional_probes.json)，SHA-256=`fce20780ae2c194b3e81880acbeb4a6c2a5d9fec352b8559487da3ccbc9e0c18`。
- step-0 norms=`||g_rec||=0.0005866184`、`||g_sem_train||=0.2936746160`、`||g_sem_smooth||=0.2829194241`；step-300 norms=`0.0005759371`、`0.2730509876`、`0.2779645008`。step-0 cosines=`C_train_smooth=+0.18470490`、`C_rec_smooth=-0.06370815`、`C_rec_train=+0.07054067`；step-300=`+0.11107681`、`-0.08296854`、`+0.04521843`。
- 对 `eps={1e-4,3e-4,1e-3}` 的 `±` symmetric probes 均在每次扰动后恢复；raw validity 全部 finite/nonconstant=`1.0/1.0`、zero-image=`0`。`-g_sem_train` 与 `-g_sem_smooth` 均改善 Smooth loss；`-g_rec` 改善 reconstruction 但恶化 Smooth，故 conflict evidence 稳定存在。
- 预定义的 mismatch 与 controllability isolation 条件均未独立成立；当前机制为“局部符号证据不能把 objective transfer mismatch、Generator controllability 与 reconstruction interaction 分离”。推荐 single repair direction=`NONE`：保持 objective、architecture 与 frozen context 不变，等待新的 bounded design decision；不自动训练、调参、续训或 promotion。
- no-update preflight=`READY`：optimizer=`false`、parameter updates=`0`、runtime restored=`true`；仅读取 TRAIN calibration=`10`，DEV/protected/test/sealed=`0`。encoder/OLS/`W_OLS`/P2/Q2/evaluators 均未改动。
- [attribution executor](../scripts/research/execute_lightgen_2d_generator_semantic_controllability_attribution_v1.py) 与 [focused test](../tests/v2/test_lightgen_2d_generator_semantic_controllability_attribution_v1.py) 已通过 compile/lint/pytest。`PROJECT MAINLINE=M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION`；`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。

## Frozen latent Generator semantic-recovery successor（2026-09-18；accepted input authority）

按用户明确授权 `AUTHORIZED_EXECUTE_FROZEN_LATENT_GENERATOR_SEMANTIC_RECOVERY_V1`，从接受版 Chen BOGT step-300 状态启动一次且仅一次 successor：只更新 `generator.phase`，冻结 encoder/OLS/scale/`T=I_100`/`W_OLS`/latent normalization/P2/Q2/evaluators，执行 `300` 个 TRAIN updates，并在 `0/100/200/300` 做完整 DEV 诊断。结果=`VALID_COMPLETED`，Outer Gate=`FROZEN_LATENT_GENERATOR_SEMANTIC_RECOVERY_V1_RESOLVED`，classification=`GENERATOR_SEMANTIC_RECOVERY_NOT_SUPPORTED`；promotion 与 generation-success claim 继续关闭。

- [successor report](../outputs/20260918_frozen_latent_generator_semantic_recovery_v1/runs/attempt_01/report.json)，SHA-256=`fcb85e17e4b3d258fbe76d975103869e8d18a11252965dd5581f02a0acdb19b3`；[preflight](../outputs/20260918_frozen_latent_generator_semantic_recovery_v1/runs/attempt_01/preflight.json)；[semantic calibration](../outputs/20260918_frozen_latent_generator_semantic_recovery_v1/runs/attempt_01/semantic_weight_calibration.json)；[step-300 checkpoint](../outputs/20260918_frozen_latent_generator_semantic_recovery_v1/runs/attempt_01/scientific_step_000300.pt)。
- 唯一授权已消费；preflight=`READY`，3-step 真实更新通过，full-runtime optimizer negative control 被拒绝；正式 optimizer 为 Adam，仅 `generator.phase`，接受版 generator moments subset 恢复。目标为 `L_recovery=beta*L_rec+gamma*L_op+delta*L_sem`，`beta=2.4509953404`、`gamma=0.5330424285`、`delta=0.0065783691`；旧 TN distribution term 未进入 objective/backward。
- 独立 Smooth DEV primary 的 macro recall=`0.1030→0.1034`，delta=`+0.0004`，paired class-stratified bootstrap `B=999`、seed=`271828` 的 95% CI=`[-0.0004,+0.0012]`，因此语义支持条件不成立。DEV reconstruction=`0.0687409→0.0673416`，delta=`-0.0013992`，`+0.005` 约束通过。
- exact accepted P2+Q2 与 real-control transfer：`R_raw=1.5288204→1.5524823`、`R_feature=3.0552277→3.2092072`；R_feature 未下降，transfer 条件不成立。successor raw CMOS finite/nonconstant=`1.0/1.0`、zero-image=`0`；光学 validity 通过。
- 本次 data reads 仅为 TRAIN=`3000` 与 DEV=`20000`；protected/test/sealed=`0`。未改变 architecture、OLS、P2/Q2、evaluators 或受保护/测试/封存数据；[executor](../scripts/research/execute_lightgen_2d_frozen_latent_generator_semantic_recovery_v1.py) 与 [focused test](../tests/v2/test_lightgen_2d_frozen_latent_generator_semantic_recovery_v1.py) 已通过 compile/lint/pytest。结论只覆盖该 frozen-latent、generator-only、300-update successor，不支持语义恢复或 promotion。
- `PROJECT MAINLINE=M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION`；输入 phase authority=`CHEN_BOGT_PHASE_COMPLETION_REPAIR_V1_RESOLVED` 保持有效；`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。不自动追加 arm、调参、续训或 promotion。

## Chen BOGT phase-completion repair（2026-09-18；accepted input authority）

本 successor 消费的 phase authority 是已接受的 frozen-model downstream propagation diagnostic：Stage 1 在 CPU 上完成 Q0/Q1/Q2/Q3 与 Q4 control 的 `B=999` within-pair calibration，Stage 2 对选中的 Q2 完成 `200` 个 selected-generated direct Generator forwards，并 exact reuse `200` 个 real-control CMOS；未训练、未 refit `B/C/L`、未改变 P2/OLS/Generator/evaluator。

- [phase repair report](../outputs/20260918_chen_bogt_phase_completion_repair_v1/runs/attempt_01/phase_completion_repair_report.json)；[phase selection](../outputs/20260918_chen_bogt_phase_completion_repair_v1/runs/attempt_01/phase_selection.json)；[selected latent bank](../outputs/20260918_chen_bogt_phase_completion_repair_v1/runs/attempt_01/selected_phase_completion_latent_bank.pt)。
- 选中 Q2=`ANCHOR-CONDITIONED_JOINT_UNCONDITIONAL`；冻结 P2+Q0 → Q2 的 `R_u=1.5095→1.5095`、`R_phi=7.1200→0.9937`、`R_z=19.0598→1.8283`、`R_raw=2.1189→1.5288`、`R_feature=7.4096→3.0552`，classification=`DOWNSTREAM_DIRECTIONALLY_IMPROVED`。
- selected raw validity 为 finite/nonconstant=`1.0/1.0`、zero-image=`0`；该 authority 只覆盖当前 frozen model/frozen bank 的 downstream propagation，不构成 generation acceptance、physical validity 或 promotion。

## Chen latent-prior support decomposition（2026-09-18；historical source authority, superseded by the compatibility calibration above）

attempt_08 的 frozen TRAIN-derived decomposition 事实保持有效：`R_TN=94.4553`、`R_independent=9.3665`、`R_copula=1.7113`、`R_real_row_control=1.2407`，描述性 marginal/dependence reduction=`0.9105`/`0.9150`；其 strict-Pareto primary diagnosis=`UNRESOLVED`、candidate=`NONE` 只表示上一轮 selector 的结果，已被本轮预先定义的 finite-sample primary compatibility rule supersede，不代表 P1/P2 工件被修改。

## Chen BOGT sampled-generation canary（2026-09-18；historical input authority, superseded by the compatibility calibration above）

此前固定 `N_LATENT=1000`、`N_OPTICAL=200` 的 canary 事实保持有效；其 primary diagnosis=`MULTI_STAGE_MISMATCH`，prior/support `R_u=85.1354`，phase `R_phi=7.1200`，full-complex-z `R_z=27.7766`，raw/feature `R_raw=2.8473`/`R_feature=8.2326`。该 canary 是本轮 P0 identity 的冻结输入，不再代表当前 latest bounded execution；其 raw CMOS validity/diversity 与无 labels 因而未计算 accuracy/recall 的事实保持不变。

## Chen BOGT training / generation-ready bundle（2026-09-18；historical, superseded by the current compatibility calibration above）

前一轮固定 `300`-update training、final TRAIN collection、wrapped-phase model fit 与 generation-ready bundle 事实保持有效；其 `VALID_COMPLETED_GENERATION_BUNDLE_READY` 只描述 bundle-ready 状态，已被上方 compatibility calibration 的 current block supersede，不再代表当前 latest execution。step-300 DEV diagnostic semantic macro recall=`0.1474` 与 Stage-A diagnostic threshold=`0.937` 的历史事实不变；本节不得覆盖上方 current block。

## 当前 research fast path / long-run resilience（2026-09-17；governance-only；historical）

本轮 Outer Gate=`RESEARCH_FAST_PATH_AND_LONG_RUN_RESILIENCE_V1_ESTABLISHED`；只建立普通 `R0/R1/R2` research 的 one-Outer-Gate/one-authorization、工程预检、heartbeat、
progress、operational recovery 和同轨迹 resume 规则；没有 scientific training/run、generation、
CUDA、evidence 或 scientific classification 变更（仅保留工程预检允许的 tiny checks）。`PROJECT MAINLINE` 仍为
`M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION`；`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。历史结果、
formal v2 compatibility、N3 scope 与 Scaled-Chen 关闭边界均保持不变。

## 4L Linear optimization-ceiling continuation（2026-09-16，VALID_COMPLETED；C3000 plateau）

按当前研究主线的 bounded continuation 方案，从已验证的 C=`4L Linear` local step `2000` 唯一续训到 `3000`，仅更新 `generator.phase`，不重训 A/B/D，不改变架构、OLS、adapter、facet、objective、数据角色或层间距。preflight attempt-01/02 均在 scientific update 前以 `invalid_harness` 停止并保留；attempt-03=`AUTHORIZABLE` 后完成唯一的 `1000` 个 CUDA updates。训练每 `200` 步保存 checkpoint、block 摘要和 progress event，C3000 endpoint 完整评估 DEV=`5000` 与 cyclic selection=`1000`，未进入 C4000。

- [continuation contract](../outputs/20260916_4l_linear_optimization_ceiling_v1/runs/4l_linear_optimization_ceiling_v1_contract_attempt_03/4l_linear_optimization_ceiling_v1_contract.v1.json)，SHA-256=`5a18f32c7e8f66566b6554f0a6adc189a93087528a40820c266c9793a68e201e`；[preflight report](../outputs/20260916_4l_linear_optimization_ceiling_v1/runs/4l_linear_optimization_ceiling_v1_preflight_attempt_03/4l_linear_optimization_ceiling_v1_preflight_report.v1.json)，SHA-256=`09a797e8155ef01dde45be656c65a7a0002f95b652260075ee854f49cbdf643c`；lineage SHA-256=`df3aadcbb65e12224e1d0ecd8ae07188c5390e39ffe4bea787757fc962723bee`。
- [aggregate report](../outputs/20260916_4l_linear_optimization_ceiling_v1/runs/4l_linear_optimization_ceiling_v1_aggregate_attempt_01/4l_linear_optimization_ceiling_v1_aggregate_report.v1.json)，SHA-256=`8687ac41431917485806ea3b3f188673544e6e6945808915db10f07212b96793`；[lineage](../outputs/20260916_4l_linear_optimization_ceiling_v1/runs/4l_linear_optimization_ceiling_v1_aggregate_attempt_01/4l_linear_optimization_ceiling_v1_aggregate_lineage.v1.json)，SHA-256=`a6eca15e88ff59851379498389a52a67ea03c2c890b13420d5c5bb3fe70ac6bc`。
- [C3000 checkpoint](../outputs/20260916_4l_linear_optimization_ceiling_v1/runs/4l_linear_optimization_ceiling_v1_c_linear_continuation_attempt_01/c_4l_linear_continuation_step_03000_checkpoint.v1.pt)，SHA-256=`b5369c6ad16391adf772ba8aa42ea9642048a52041b07d50ad0e3c4928cc6270c`；[sufficient observations](../outputs/20260916_4l_linear_optimization_ceiling_v1/runs/4l_linear_optimization_ceiling_v1_c_linear_continuation_attempt_01/c_4l_linear_continuation_step_03000_sufficient_observations.v1.pt)，SHA-256=`e2072154d88fa423e988db0968038b5a67da5e1d7b149be6c3a6792019bb9a3ac`。

| endpoint | Smooth macro recall | DEV `L_image` | raw CMOS mean power | raw/cyclic |
| --- | ---: | ---: | ---: | --- |
| C2000 parent | 0.5774 | 0.0467247652 | 7606.3509 | pass / pass |
| C3000 continuation | 0.5790 | 0.0469439335 | 7628.1403 | pass / pass |

主比较 `Delta_R(C3000−C2000)=+0.0016`，paired bootstrap（`999` replicates，seed=`271828`）95% CI=`[-0.0002,+0.0034]`，`p_positive=0.95095`；图像损失变化为 `+0.0002191683`。预注册 C3000 条件中，`Delta_R>=+0.010`=`false`、CI lower `>0`=`false`，reconstruction/raw validity/cyclic transfer=`true`，所以 classification=`4L_LINEAR_CONTINUATION_PLATEAU_SUPPORTED`，`continue_to_4000=false`。这支持“在当前固定 generator-only continuation 与优化器状态下已进入平台”，不等价于证明所有优化方法都已达到全局最优。

C2000→C3000 各类 recall 为 `0:0.952→0.952`、`1:0→0`、`2:0.698→0.700`、`3:0.714→0.716`、`4:0.810→0.814`、`5:0.078→0.086`、`6:0.676→0.686`、`7:0.524→0.516`、`8:0.936→0.936`、`9:0.386→0.384`；改善主要来自 5/6 类，7/9 类略降，1 类仍为零。C3000 相对历史 2L reference `0.6170` 仍低 `0.0380`，该差值仅作描述性参考，不能作 4L 优越性结论。

续训参数固定为：4 层 generator、层间距 `3 mm`、production grid=`2048`、linear facet（`tau=0`）冻结、rank-9 adapter/2 层 encoder/complex 100-port OLS/`W_OLS` 冻结；`L=L_anchor+28.48464149920509·L_transfer+13.353282014834436·L_image`；Adam `betas=(0.9,0.999)`、`eps=1e-8`、weight decay=`0`、AMSGrad=`false`，恢复 C2000 moments；local `2000→3000` 学习率 `5e-5→3e-5`；balanced multi-epoch sampler seed=`1234`、logical batch=`10`、microbatch=`2`，恢复 parent cursor=`2000`。preflight 三步真实更新、冻结组 negative control、sampler/RNG interrupted-resume exact match 均通过；RTX 3060 Laptop 实际续训约 `924.07 s`，peak VRAM `1,451,138,048` bytes。

本次只产生 C 线 continuation evidence；未执行 ReLU 训练、Stage B、sampled generation、physical Te validation、protected/official-test/sealed read 或 formal promotion。半小时监控已在实验完成后关闭。后续若要研究 Te 薄膜的实际参数，应另行定义材料参数与新的 bounded contract，不能把本次 ideal linear continuation 当作 physical Te 结论。

## Strict TRAIN-calibrated ReLU90 successor（2026-09-15，VALID_COMPLETED；Outer Gate NOT_SUPPORTED；A 复用 + B=2000）

按用户明确授权 `EXECUTE_STRICT_TRAIN_CALIBRATED_RELU90_GAIN_V2`，在
`research/lightgen-ols-native-semantic-generation-v1`、HEAD=`a9cd1749adfa690e282fd52a6425564fc2cf102c` 上完成一次且仅一次有效 successor scientific scope。attempt-01/02 在 scientific update 前因新 harness 校验错误停止并保留；attempt-03 的 preflight=`AUTHORIZABLE`，随后只训练 B 的 `2000` 个 CUDA updates。既有 nominal ideal-facet A 端点通过完整 exact-reuse audit，未重训；未读取 protected、official-test 或 sealed，未生成图像/后处理图，也未改变旧 nominal outputs。

- contract authority=`DEFINE_STRICT_TRAIN_CALIBRATED_RELU90_GAIN_V2`；[contract](../outputs/20260915_strict_train_calibrated_relu90_gain_v2/runs/strict_train_calibrated_relu90_gain_v2_contract_attempt_03/strict_train_calibrated_relu90_gain_v2_contract.v2.json)，SHA-256=`b4488885997f65ad21a2c34c97613f25d3dd9a82f3d95c63ec0838d5537bfb1b`。
- [preflight report](../outputs/20260915_strict_train_calibrated_relu90_gain_v2/runs/strict_train_calibrated_relu90_gain_v2_preflight_attempt_03/strict_train_calibrated_relu90_gain_v2_preflight_report.v2.json)，SHA-256=`14ae74b7f7a0dedc7da8261b4bb475d758b46fc3202a78fa4d883d0c00dc71de`；严格 TRAIN calibration 为 `1000=10×100`，disjoint TRAIN validation 为 `1000=10×100`，`tau=0.10682560942718147`，`eta_cal=0.9000000000`，`eta_train_valid=0.9015109995`；3-step real update、stage-2 adapter freeze、negative control、sampler/RNG resume 均通过。
- A authority 为上一轮 [nominal A report](../outputs/20260915_ideal_facet_relu_long/runs/lightgen_2d_ideal_facet_relu_long_a_linear_attempt_05_minimal_ondemand/lightgen_2d_ideal_facet_relu_long_a_linear_scientific_report.v1.json)，reuse audit=`PASS/REUSED_VALID_AUTHORITY`；A=`R0 0.5526 → R2000 0.6170`，`L_image 0.05241717 → 0.04578155`。
- B [scientific report](../outputs/20260915_strict_train_calibrated_relu90_gain_v2/runs/strict_train_calibrated_relu90_gain_v2_b_relu90_attempt_03/strict_train_calibrated_relu90_gain_v2_b_relu90_scientific_report.v2.json)，SHA-256=`51865886651d5a92a5ee6b2333750bd59066954f4c99bf423d557cbe93aa74a4`；B=`R0 0.5454 → R2000 0.6138`，`L_image 0.04967835 → 0.04630363`。
- [aggregate report](../outputs/20260915_strict_train_calibrated_relu90_gain_v2/runs/strict_train_calibrated_relu90_gain_v2_aggregate_attempt_03/strict_train_calibrated_relu90_gain_v2_aggregate_report.v2.json)，SHA-256=`def0a60c7176b82bbb776b4c6989da8185b744c5d4504b1eec1995b9d4344ca7`；`VALID_COMPLETED`，classification=`STRICT_TRAIN_CALIBRATED_RELU90_NOT_SUPPORTED`。

主比较为 `Delta_R(B2000−A2000)=-0.0032`，paired bootstrap 95% CI=`[-0.00641,0]`，不满足预注册 `Delta_R>=+0.05` 且 CI lower `>0`；`Delta_L_image=+0.00052208` 满足重建约束，B cyclic template/Smooth transfer 与 raw validity 均通过。因此严格 TRAIN 标定并未支持 ReLU90 对语义生成的增益。B 的描述性机制量为 facet power retention `eta0=0.94923530 → eta2000=0.95656532`、active fraction `0.57471181 → 0.60559027`、suppressed fraction `0.42528817 → 0.39440971`；tau 在训练中冻结，facet 不可训练。

独立 Stage-A firewall 使用两臂最高 `R=0.6170<0.937`，状态仍为 `STAGE_A_FIREWALL_FAILED`、`stage_b_authorized=false`。本次没有 rank/loss/lambda/lr/geometry sweep、额外 arm、retry、Stage B、latent prior、sampled generation、physical Te validation 或 formal promotion；strict successor 结果不改写前一轮 nominal `LONG_IDEAL_FACET_RELU90_NOT_SUPPORTED`，也不构成真实 Te 薄膜材料验证。

## N6 nonlinear fiber-facet activation canary（2026-09-14，VALID_COMPLETED；Outer Gate NOT_SUPPORTED；one-shot A/B）

N5-V2 已作为线性 optical-depth 主线的终止实验：`OPTIMIZATION_HORIZON_DOMINANT_TRUE_DEPTH_NOT_SUPPORTED`，不再追加线性深度候选。按用户明确授权 `EXECUTE_N6_NONLINEAR_FIBER_FACET_ACTIVATION_CANARY_V1`，在
`research/lightgen-ols-native-semantic-generation-v1`、HEAD=`a9cd1749adfa690e282fd52a6425564fc2cf102c` 上，使用 exact N5-A step-300 endpoint（global step=`600`）完成一次且仅一次 A/B CUDA scientific scope，各 `300` updates，global step=`600→900`。A 为 linear continuation control；B 在 canonical `lightgen_generator_entrance` facet 加入固定 algorithmic nonlinear surrogate；两臂均保持 rank=`9`、encoder/OLS/`W_OLS` 冻结，仅 adapter 与现有 generator phase 更新。该 surrogate 不是实测材料曲线，结果不构成 physical material validation。

- design authority=`DEFINE_N6_NONLINEAR_FIBER_FACET_ACTIVATION_CANARY_V1`；[contract](../outputs/20260914_n6/runs/lightgen_2d_nonlinear_fiber_facet_activation_canary_contract_v1_attempt_02/lightgen_2d_nonlinear_fiber_facet_activation_canary_contract.v1.json)，SHA-256=`1943c9327076e57451923ef3d0bc6815b9f336a5e7fc8efc2b6348702192bfd5`。
- bounded preflight attempt-01/02/03 为 `INVALID_HARNESS`，均在 scientific update/forward 前停止并保留；attempt-04=`AUTHORIZABLE`，[report](../outputs/20260914_n6/runs/lightgen_2d_nonlinear_fiber_facet_activation_canary_preflight_v1_attempt_04/lightgen_2d_nonlinear_fiber_facet_activation_canary_preflight_report.v1.json) SHA-256=`b16742d3b3805293ff2cc8a06d92b32c494cab0e07837fa22056cac55f634aa6`，lineage SHA-256=`7d1c63c360a45a813575aeb3b5a3d47d265073f85f86e10154007a6ef983f07c`；preflight scientific updates/forward evaluations=`0`。
- A [report](../outputs/20260914_n6/runs/lightgen_2d_nonlinear_fiber_facet_activation_canary_a_attempt_01/lightgen_2d_nonlinear_fiber_facet_activation_canary_a_linear_continuation_control_scientific_report.v1.json) SHA-256=`7ec8e7c0d02f46cef6ac36dbaa24cf1a8a70a8ebbbce05a7057a76501510bc8c`；lineage SHA-256=`9092f183040c42f994b8c0e2aef4594e80d7931bc7c512acb76c3719d6e030ec`。
- B [report](../outputs/20260914_n6/runs/lightgen_2d_nonlinear_fiber_facet_activation_canary_b_attempt_01/lightgen_2d_nonlinear_fiber_facet_activation_canary_b_nonlinear_facet_scientific_report.v1.json) SHA-256=`8763b1547f8f9a4db102f0d987bca8ac1c76048cb688aaf6031e52b1784233c7`；lineage SHA-256=`8dc0745a05fce9d30f33ece0e6e69fa54f16ef12cb45c626d982f3eb1a68da34`。
- [N6 aggregate report](../outputs/20260914_n6/runs/lightgen_2d_nonlinear_fiber_facet_activation_canary_aggregate_v1_attempt_01/lightgen_2d_nonlinear_fiber_facet_activation_canary_aggregate_report.v1.json) SHA-256=`56534572368253dd0c488a695e31050a240a4b3e09535c8241eeff4c4a676ce7`；[lineage](../outputs/20260914_n6/runs/lightgen_2d_nonlinear_fiber_facet_activation_canary_aggregate_v1_attempt_01/lightgen_2d_nonlinear_fiber_facet_activation_canary_aggregate_lineage.v1.json) SHA-256=`ad31df72f6edc00d8d6ad8e8d6166f7160c614ab5d270f0aee1d9a7cfa7a560b`；`VALID_COMPLETED`，Outer Gate=`N6_NONLINEAR_FIBER_FACET_ACTIVATION_NOT_SUPPORTED`。

| candidate | Smooth 0/100/200/300 | DEV `L_image` 0→300 | `R300` | facet step-300 | cyclic/raw |
| --- | --- | ---: | ---: | --- | --- |
| A linear control | 0.5526 / 0.5518 / 0.5630 / 0.5788 | 0.05241717→0.04968191（Δ=-0.00273526） | 0.5788 | identity；suppressed=0；power ratio=1.0000 | pass / pass |
| B nonlinear facet | 0.5514 / 0.5530 / 0.5632 / 0.5800 | 0.05093504→0.04988193（Δ=-0.00105311） | 0.5800 | activation=0.66219；suppressed=0.33781；power ratio=0.97556 | pass / pass |

主比较 `Delta_NL=R_B300-R_A300=+0.0012`，paired bootstrap 95% CI=`[-0.0014,+0.0040]`；未达到预注册的 `>=+0.05` 且 CI lower `>0`，故 nonlinear facet activation 未支持。B reconstruction 约束通过（`-0.00105311 <= +0.005`），两臂 cyclic template/Smooth transfer、raw validity 与 diversity 记录均通过。step-300 adapter residual Fro 为 A=`6.02531633`、B=`6.09274193`；factor-product 前 9 个 singular values 为 A=`[4.366102,3.237813,2.908985,1.908045,1.542328,1.053501,1.023655,1.002686,1.000875]`、B=`[4.439224,3.274072,2.928571,1.927012,1.532621,1.049934,1.023700,1.002661,1.000842]`；generator phase-change Fro 为 A=`9.11801245`、B=`8.84950819`。

独立 Stage-A firewall 保持 `0.937`：A=`0.5788<0.937`、B=`0.5800<0.937`，均为 `STAGE_A_FIREWALL_FAILED`、`stage_b_authorized=false`。本次未改变 optical depth、rank、loss、lambda 或 learning rate；没有 retry、Stage B、generation、protected/test/sealed read 或 formal promotion。N6 是 bounded scientific outcome，不授权任何下游推进。

## N5 V2 true-optical-depth comparison（2026-09-14，VALID_COMPLETED；one-shot A/B）

按用户明确授权 `EXECUTE_N5_SUCCESSOR_TRUE_OPTICAL_DEPTH_COMPARISON_V2`，在
`research/lightgen-ols-native-semantic-generation-v1`、HEAD=`a9cd1749adfa690e282fd52a6425564fc2cf102c`
上完成一次且仅一次 V2 bounded preflight、A/B 各 `300` 个 CUDA updates 与冻结 paired
aggregate。V2 是 V1 `INVALID_N5_SCIENTIFIC_UNEXPECTED_EXECUTION_ERROR` 的独立 successor，
不是 retry/continuation；V1 scientific metrics 与未持久化 observations 均未消费。V2 继续使用
rank-9 OLS interface、冻结 encoder/OLS/`W_OLS`、同一 C1 step-300 起点、相同 objective、
optimizer、数据顺序与预算；仅 B 加入 full-2048-grid 的 `0.75 mm` / midpoint phase / `0.75 mm`
factorization。没有 rank/loss/lambda/lr 调整、retry、额外 arm 或下游阶段。

- V2 contract authority=`DEFINE_N5_SUCCESSOR_TRUE_OPTICAL_DEPTH_COMPARISON_V2_CONTRACT`；
  [contract](../outputs/20260914_n5_v2/runs/lightgen_2d_true_optical_depth_comparison_v2_contract_v1_attempt_01/lightgen_2d_true_optical_depth_comparison_v2_contract.v1.json)，SHA-256=`f7928d640093d8133380233c0ee451221b93f1ee03236b381227228a5c1df9fb`。
- preflight attempt-03 [report](../outputs/20260914_n5_v2/runs/lightgen_2d_true_optical_depth_comparison_v2_preflight_v1_attempt_03/lightgen_2d_true_optical_depth_comparison_v2_preflight_report.v1.json)，SHA-256=`86933442d9051d6a11210b08c14cb59b9c9e1f2f229d2feaa75f0ed2a2d11992`；lineage SHA-256=`8a0f169fa41edceca19ffd271118b55cc5455badd6e24772aa4849665218d519`；`AUTHORIZABLE`，scientific updates=`0`。此前 V2 preflight attempt-01/02 的 harness INVALID artifacts 保留且未覆盖。
- A [report](../outputs/20260914_n5_v2/runs/lightgen_2d_true_optical_depth_comparison_v2_candidate_a_attempt_01/lightgen_2d_true_optical_depth_comparison_v2_a_continued_c1_scientific_report.v1.json)，SHA-256=`ed796d14971aa8cf44f708ba3dc274601f38db46b82559dc4572bcd202d14fca`；lineage SHA-256=`a77fb0b58afc6f8d0a53c45cc68f362051f1a153abc119b413183cdab70361f8`；`VALID_COMPLETED`。
- B [report](../outputs/20260914_n5_v2/runs/lightgen_2d_true_optical_depth_comparison_v2_candidate_b_attempt_01/lightgen_2d_true_optical_depth_comparison_v2_b_true_depth_c1_scientific_report.v1.json)，SHA-256=`8f3bc38c927a64f97431325504860bd71c622cd989e248022fe736706c685681`；lineage SHA-256=`d092f47bef7021c9bae73423a4a8af19b54b1449ac448beebca691265dfe91e5`；`VALID_COMPLETED`。
- [V2 aggregate report](../outputs/20260914_n5_v2/runs/lightgen_2d_true_optical_depth_comparison_v2_aggregate_v1_attempt_01/lightgen_2d_true_optical_depth_comparison_v2_aggregate_report.v1.json)，SHA-256=`78fc9061dacac1f68180b2d4619ca35cb3356b7c4b007cfe37f21a8092cfd002`；[lineage](../outputs/20260914_n5_v2/runs/lightgen_2d_true_optical_depth_comparison_v2_aggregate_v1_attempt_01/lightgen_2d_true_optical_depth_comparison_v2_aggregate_lineage.v1.json)，SHA-256=`1662dabded7fda151bfb017c45b73de89cf75b745fd296c7b4074a770a8a8e67`；classification=`OPTIMIZATION_HORIZON_DOMINANT_TRUE_DEPTH_NOT_SUPPORTED`，Outer Gate=`N5_V2_TRUE_OPTICAL_DEPTH_COMPARISON_RESOLVED`。

| candidate | Smooth 0/100/200/300 | `R300-0.4838` | paired 95% CI | DEV image delta | cyclic/raw |
| --- | --- | ---: | --- | ---: | --- |
| A continued C1 | 0.4838 / 0.5254 / 0.5294 / 0.5526 | +0.0688 | [+0.0612,+0.0768] | +0.00256892 | pass / pass |
| B true depth C1 | 0.4838 / 0.5240 / 0.5278 / 0.5504 | +0.0666 | [+0.0582,+0.0746] | +0.00249731 | pass / pass |

主比较为 `Delta_depth=R_B300-R_A300=-0.0022`，paired bootstrap 95% CI=`[-0.0042,-0.0002]`，
未达到预注册的 `>=+0.05` 且 CI lower `>0`，因此 true-depth midpoint 在本次固定 `300`-update
窗口内不支持相对 continued C1 的语义增益。两臂 reconstruction、cyclic template/Smooth transfer
与 raw validity 均通过；这是真实 scientific Gate 失败而非 harness INVALID。A 仅训练 adapter 与
existing generator phase（`23,600` real parameters）；B 另训练 `10,000` 个 midpoint depth-phase
参数（总计 `33,600`），full `2048x2048`、无 midpoint crop/pad/resample。step-300 adapter residual
Fro 为 A=`5.39799668`、B=`5.40529312`；generator phase-change Fro 为 A=`12.30389625`、
B=`11.55147097`；B midpoint phase-change Fro=`7.82456298`。

独立 Stage-A firewall 仍失败：A=`0.5526<0.937`、B=`0.5504<0.937`，两者均
`STAGE_A_FIREWALL_FAILED`、`stage_b_authorized=false`。本次实际更新仅为声明的 adapter/generator
(B 含 midpoint phase)；encoder、OLS、`W_OLS` 未更新。Stage B、scale/prior、generation、
protected/test/sealed read 与 formal promotion 均未开放。原 N4 aggregate 仍 unresolved，V1
INVALID 仍为永久历史边界；V2 结果不改写二者。

## N4 OLS-interface/synthesis model-family comparison（2026-09-13/14，INVALID；C1 one-shot harness stop，C2 valid candidate only）

按用户明确授权 `EXECUTE_N4_OLS_INTERFACE_SYNTHESIS_MODEL_FAMILY_COMPARISON_V1`，在
`research/lightgen-ols-native-semantic-generation-v1`、HEAD=`a9cd1749adfa690e282fd52a6425564fc2cf102c`
上完成 N4 contract、bounded preflight，以及 C1/C2 各自唯一一次的 scientific scope。C0
沿用 V11 rank-9 endpoint；C1 为 rank-9 adapter 加可训练的当前 2-layer generator；C2
为 rank-9 adapter 加当前 2-layer propagation stack 与一个可训练的 co-planar entrance
phase screen，encoder、OLS、`W_OLS`、Smooth evaluator 与数据角色冻结。没有 rank/loss/
lambda/lr 调参、retry、Stage B、generation、protected/test/sealed read 或 promotion。

- semantic authority=`DEFINE_N4_OLS_INTERFACE_SYNTHESIS_MODEL_FAMILY_CONTRACT_V1`；continuation contract [contract](../outputs/20260913_n4/runs/lightgen_2d_ols_interface_synthesis_model_family_contract_v1_attempt_05/lightgen_2d_ols_interface_synthesis_model_family_contract.v1.json)，SHA-256=`ae3b90bb84e7748094217ff04ef8e9141c9dd1ab5e89a21376efc73978ee2cd3`。
- initial bounded preflight attempt-04 与 C1 停止后的 continuation preflight attempt-05 均为 `AUTHORIZABLE`；attempt-05 [report](../outputs/20260913_n4/runs/lightgen_2d_ols_interface_synthesis_model_family_preflight_n4_attempt_05/lightgen_2d_n4_model_family_preflight_report.v1.json)，SHA-256=`3fec74d08145d5b368647d94cf9ceb69754a8f7c73abceed51f3ebbafc530510`；[lineage](../outputs/20260913_n4/runs/lightgen_2d_ols_interface_synthesis_model_family_preflight_n4_attempt_05/lightgen_2d_n4_model_family_preflight_lineage.v1.json)，SHA-256=`f12828b00874e997a675473c62904007157c369f472249226453eb6597e6bdb7`。attempt-05 只做 bounded continuation preflight，未重复 C1 scientific run。
- C1 [report](../outputs/20260913_n4/runs/lightgen_2d_ols_interface_synthesis_model_family_c1_attempt_01/lightgen_2d_n4_c1_scientific_report.v1.json)，SHA-256=`2b6b8c9d4b5bbe2c48eab18ee0d855c95f699cc0abeb93202bfe6340e51c6e61`；结果=`INVALID/invalid_harness`，classification=`INVALID_N4_SCIENTIFIC_UNEXPECTED_EXECUTION_ERROR`。C1 已完成 `300` 个 optimizer updates，但末端 checkpoint/resume probe 在 scientific consumption 后错误请求已耗尽 sampler 的下一批，触发 `V2 train sampler is exhausted`；C1 scientific metrics 不被接受，且按 one-shot 规则不重跑。
- C2 [report](../outputs/20260913_n4/runs/lightgen_2d_ols_interface_synthesis_model_family_c2_attempt_01/lightgen_2d_n4_c2_scientific_report.v1.json)，SHA-256=`660454787c3df9d5b6aa69facfd9ec94acc67c19ac08cff1e23cdcf754b81003`；[lineage](../outputs/20260913_n4/runs/lightgen_2d_ols_interface_synthesis_model_family_c2_attempt_01/lightgen_2d_n4_c2_scientific_lineage.v1.json)，SHA-256=`21c6840d62012a9fa7a9cca6ffd2cc377aa3be3e2832223f5aead1499763edbd`；结果=`VALID_COMPLETED`，分类=`N4_CANDIDATE_ADEQUATE`。C2 的 `33,600` 个声明参数可训练，encoder/OLS/`W_OLS` 保持冻结。

| C2 step | Smooth macro | DEV L_image |
| ---: | ---: | ---: |
| 0 | 0.3056 | 0.0531686321 |
| 100 | 0.3624 | 0.0528490431 |
| 200 | 0.4228 | 0.0472207439 |
| 300 | 0.4844 | 0.0487769855 |

C2 相对 C0 的 Smooth delta=`+0.1788`，paired bootstrap 95% CI=`[+0.16759,+0.18980]`；
相对候选 step-0 的 DEV image delta=`-0.00439165`。cyclic template transfer、cyclic
Smooth transfer、broad/not-pair-dominated 与 raw validity 均通过，故 C2 单臂为 bounded
candidate evidence。其 adapter residual step-300 Fro=`4.55118448`、spectral=`3.59413600`、
诊断 numerical rank=`100`；factor-product 前 9 个 singular values 从 `3.59413600` 到
`0.000335709`，理论 factor width 仍为 `9`。

由于 C1 没有有效 scientific report，N4 的对称比较 aggregate 未生成，不能正式选择 C1
或 C2，也不能建立 `N4_OLS_INTERFACE_SYNTHESIS_COMPARISON_RESOLVED`。脚本已修正末端
sampler 边界并经 continuation preflight 消费，但该修复不追溯改变 C1 immutable report。
C2 的独立 Stage-A firewall 仍失败：`R300=0.4844 < 0.937`；因此 Stage B、generation、
scale/prior、protected/test/sealed read 与 formal promotion 均关闭。

## N3 low-rank model-order selection（2026-09-12，VALID_COMPLETED；一次性候选执行）

按用户明确授权 `授权执行 N3 model-order selection`，在
`research/lightgen-ols-native-semantic-generation-v1`、HEAD=`a9cd1749adfa690e282fd52a6425564fc2cf102c`
上完成 N3 contract、bounded preflight 与候选模型阶选择。N3 只比较既有 V11
rank-9 机制的嵌套 rank=`1..9`：rank=`1..8` 各执行唯一一次 `300`-update CUDA
scientific arm，rank=`9` 在完成精确 V11 等价性核验后复用既有有效 authority；没有
rank=0、rank>9、adaptive expansion、second arm、调参或下游阶段。

- [N3 contract](../outputs/20260912_n3/runs/n3_low_rank_model_order_contract_v1_attempt_01/n3_low_rank_model_order_contract.v1.json)，SHA-256=`d2b72ce978e4f28fdbb89441a0e84bc7e7792726def310beac129e94b9b4b181`；semantic authority=`DEFINE_N3_LOW_RANK_MODEL_ORDER_CONTRACT_V1`，父语义 authority=`DEFINE_LIGHTGEN_2D_LOW_RANK_SEMANTIC_RECONSTRUCTION_COMPATIBILITY_CONTRACT_V11`。
- bounded preflight attempt-01/02 均为 `INVALID_HARNESS`，scientific=`0`，保留且未重写；修正 N3 rank-specific gradient probe 后 attempt-03=`AUTHORIZABLE`：[report](../outputs/20260912_n3/runs/n3_low_rank_model_order_preflight_v1_attempt_03/n3_low_rank_model_order_preflight_report.v1.json)，SHA-256=`cb77e170cb3814533df9c73c1571cbd50e548527d82ed6ecf650ed8d32345af8`；[lineage](../outputs/20260912_n3/runs/n3_low_rank_model_order_preflight_v1_attempt_03/n3_low_rank_model_order_preflight_lineage.v1.json)，SHA-256=`7e9e1502455c62096f71f6dd4c0615c4c192f9e87e721afa6ca48173794b323e`。
- [aggregate report](../outputs/20260912_n3/runs/n3_low_rank_model_order_aggregate_v1_attempt_01/n3_low_rank_model_order_aggregate.v1.json)，SHA-256=`1b6bd6af18462f717fa19532e1a981e872a4e48fe6498ed561890f6570ae89de`；[lineage](../outputs/20260912_n3/runs/n3_low_rank_model_order_aggregate_v1_attempt_01/n3_low_rank_model_order_aggregate_lineage.v1.json)，SHA-256=`93c3d4ffcfad5e68b435d7063e6e6c834b700cde2e4b3ccd95391f3bee8b8a8f`；`VALID_COMPLETED`，Outer Gate=`N3_LOW_RANK_MODEL_ORDER_RESOLVED`。

冻结条件：`T_r=I_100+U_rV_r^H`，nested initialization `U0=0.01*[e0..e_(r-1)]`、
`V0=0`，V11 semantic/image objective 与 `lambda_image=13.353282014834436` 保持固定；
仅 adapter 的 U/V trainable，encoder、generator、OLS、`W_OLS`、数据角色、optimizer、
seed、batch 与 300-update budget 保持不变。attempt-03 对 rank=`1..8` 的 gradient
均有限且可用，rank=`9` reuse=`REUSED_VALID_AUTHORITY`。首次 controller 在 rank-1
有效报告写入后仅因 lineage assembly 字段读取错误退出；修正消费路径后读取同一完整
rank-1 报告并继续 rank=`2..8`，没有重跑 rank-1。

| rank | DEV Smooth R300 | Smooth Delta | paired CI lower | DEV image Delta | cyclic transfer | candidate |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 0.2464 | +0.1428 | +0.13559 | -0.011420 | 10/10；broad | adequate |
| 2 | 0.2504 | +0.1468 | +0.13960 | -0.011015 | 10/10；broad | adequate |
| 3 | 0.2518 | +0.1482 | +0.14100 | -0.010788 | 10/10；broad | adequate |
| 4 | 0.2534 | +0.1498 | +0.14219 | -0.011470 | 10/10；broad | adequate |
| 5 | 0.2572 | +0.1536 | +0.14600 | -0.011971 | 10/10；broad | adequate |
| 6 | 0.2614 | +0.1578 | +0.15000 | -0.012548 | 10/10；broad | adequate |
| 7 | 0.2668 | +0.1632 | +0.15540 | -0.013109 | 10/10；broad | adequate |
| 8 | 0.2760 | +0.1724 | +0.16420 | -0.013594 | 10/10；broad | adequate |
| 9（V11 reuse） | 0.3056 | +0.2020 | +0.19360 | -0.013893 | 10/10；broad | adequate |

候选 adequacy 以预注册的 `Smooth Delta>=+0.05`、paired CI lower `>0`、DEV image
Delta `<=+0.005`、known cyclic template/Smooth transfer、raw validity 全部通过定义。
因此 `model_adequate_ranks=[1..9]`，按“minimum adequate integer rank”规则选择
`r*=1`；没有 complexity penalty，也没有按最大 Smooth 选择。rank 增大带来更高的
描述性 Smooth 与 residual flexibility，但不改变本次 model-order 规则。

Stage-A firewall 独立保持 `DEV Smooth R300>=0.937`：所有候选均失败，最佳 rank-9
也只有 `0.3056<0.937`，故状态为 `STAGE_A_FIREWALL_FAILED`。本次不开放 Stage B、
scale/prior、generation、protected/test/sealed read 或 formal promotion；N3 结果不
等同于生成就绪或 Stage-A eligibility。

## V11 low-rank semantic/reconstruction compatibility（2026-09-12，VALID_COMPLETED；一次且仅一次）

按用户明确授权 `EXECUTE_LIGHTGEN_2D_LOW_RANK_SEMANTIC_RECONSTRUCTION_COMPATIBILITY_V11`，在
`research/lightgen-ols-native-semantic-generation-v1`、HEAD=`a9cd1749adfa690e282fd52a6425564fc2cf102c`
上完成 V11 contract、bounded preflight 与唯一一次 `300`-update CUDA scientific run。V11
只在 V10.1 rank-9 OLS-to-generator remapping 上加入固定的 reconstruction compatibility
term；没有 rank search、second arm、lambda/lr tuning 或任何下游阶段。

- [V11 contract](../outputs/20260912_v11/runs/lightgen_2d_low_rank_semantic_reconstruction_compatibility_contract_v11_attempt_01/lightgen_2d_low_rank_semantic_reconstruction_compatibility_contract.v11.json)，SHA-256=`6c49a0cff4e08622351438529cebafe9388537d713a5b78b9fabbc953904efff`；semantic authority=`DEFINE_LIGHTGEN_2D_LOW_RANK_SEMANTIC_RECONSTRUCTION_COMPATIBILITY_CONTRACT_V11`，supersedes V10.1 contract `45210d92...ab3d` only for the added image-compatibility objective.
- [preflight report](../outputs/20260912_v11/runs/lightgen_2d_low_rank_semantic_reconstruction_compatibility_preflight_v11_attempt_01/lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_preflight_report.v1.json)，SHA-256=`b5078cdd61664f447ab880ad32f57b6f0d4287ea88f446c111414560f6743386`；[lineage](../outputs/20260912_v11/runs/lightgen_2d_low_rank_semantic_reconstruction_compatibility_preflight_v11_attempt_01/lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_preflight_lineage.v1.json)，SHA-256=`4910a4de5b928f97c4c1ecf9af178ad66e3cf54ad94e22f5899a4f3f9fe4f695`；`AUTHORIZABLE`，scientific=`0`。
- [scientific input](../outputs/20260912_v11/runs/lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_attempt_01/lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_input.v1.json)，SHA-256=`b0c5c6a80f0cd58be385762f5bf17b703e0f790a76aefb91586705874601b37a`；[report](../outputs/20260912_v11/runs/lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_attempt_01/lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_report.v1.json)，SHA-256=`bc603b7a3b7ef1590bc354cb0b62464f3ececcb2390834d864d58bfbe9158184`；[lineage](../outputs/20260912_v11/runs/lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_attempt_01/lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_lineage.v1.json)，SHA-256=`2c7d7c99a48c6a5a60d6524a38aa9facd6b7e0203138689f3830dcc7308ac9c8`。
- [sufficient observations](../outputs/20260912_v11/runs/lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_attempt_01/lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_sufficient_observations.v1.pt)，SHA-256=`2c4e623ce93c00dd8f865366cf7e4b3acfcea2bb203ac8f0e6b15532e61e7da9`；[step-300 cyclic raw outputs](../outputs/20260912_v11/runs/lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_attempt_01/lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_step300_cyclic_raw_outputs.v1.pt)，SHA-256=`46a0c34f94ef144dfca75b42139fcfe0108b8a99fb2e6133dd7edfe4abbbf050`；reload=`PASS`。

冻结协议：`T_adapt=I_100+UV^H`、factor width/rank=`9`、`U0=0.01*I_9`、`V0=0`；exact V3 CONTROL encoder、exact V4 generator、100-port complex OLS、`W_OLS`、TRAIN-only V5 prototypes、historical TRAIN templates 与 SmoothMNISTEvaluator v2.1 全部冻结。仅 adapter `u_real/u_imag/v_real/v_imag` 可训练，共 `3,600` 个实参数。目标为
`L_V11=L_sem+lambda_image*L_image`，其中
`L_sem=L_anchor+28.48464149920509*L_transfer`；使用同一初始 TRAIN batch 在 step-0 冻结
`g_sem=0.0016925142105153586`、`g_image=0.00012674892923566114`、
`lambda_image=13.353282014834436`，无 clipping、sweep、tuning 或 recalibration。Adam，
`lr=1e-3`，batch=`10`，grid=`2048`，评估/检查点=`0/100/200/300`；设备为 RTX 3060
Laptop、PyTorch=`2.5.1`、CUDA build=`12.1`。

| step | TRAIN L_sem | TRAIN L_image | TRAIN L_total | DEV L_image | Smooth macro | template-anchor macro | correct-template margin | OLS margin mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 21.98033819 | 0.07510780 | 22.98327384 | 0.06706161 | 0.1036 | 0.1706 | -0.00810254 | 0.00208249 |
| 100 | 21.17238071 | 0.06292281 | 22.01260677 | 0.05742796 | 0.2488 | 0.5156 | 0.00154549 | 0.00208249 |
| 200 | 21.03977135 | 0.05001093 | 21.70758137 | 0.05274165 | 0.2478 | 0.5530 | 0.00209791 | 0.00208249 |
| 300 | 20.83108337 | 0.05423452 | 21.55529220 | 0.05316863 | 0.3056 | 0.5980 | 0.00338574 | 0.00208249 |

V11 Outer Gate=`V11_LOW_RANK_SEMANTIC_RECONSTRUCTION_COMPATIBILITY_SUPPORTED`，五项 criteria 全部通过：natural Smooth improvement（`Delta_R=+0.2020`，paired bootstrap 95% CI=`[+0.1936,+0.2102]`）、DEV reconstruction compatibility（`Delta_L_image=-0.013893 <= +0.005`）、known cyclic template transfer、known cyclic Smooth transfer 与 raw validity。cyclic 结果为 template delta mean=`+0.05129368`、95% CI=`[+0.05060920,+0.05198676]`，Smooth delta mean=`+9.92621360`、95% CI=`[+9.72556023,+10.11589997]`，两者均 `10/10` positive pair；base/shifted raw 均 finite、nonconstant、structured，shape=`[1000,1,200,200]`。DEV step-0/300 raw 均 finite、nonconstant、structured，且 exact duplicate=`0`、unique output=`5000`。

adapter residual step-0=`0`；step-300 residual Frobenius=`3.02419035`、spectral=`2.67642540`、max-abs=`0.17817893`。`UV^H` 前 9 个 singular values 为 `2.67642540, 1.40705576, 0.05017205, 0.01090476, 0.00563522, 0.00061744, 0.00037506, 0.00014226, 0.00008291`；其余约为 `1e-7` 的 float32 identity-subtraction noise。故默认 numerical rank=`100` 仅为诊断值，参数化的理论 residual rank 仍为 `9`。

独立 Stage-A firewall 仍失败：DEV step-300 Smooth macro=`0.3056 < 0.937`，状态=`STAGE_A_FIREWALL_FAILED`；因此 Stage B、scale/prior、generation、protected/test/sealed read 与 formal promotion 均未开放。边界确认：adapter update=`YES`，encoder update=`NO`，generator update=`NO`，rank sweep=`NO`，retry=`NO`，Stage B=`NO`，generation=`NO`，protected/test/sealed=`NO`，promotion=`NO`；training authorization、generation readiness 与 formal promotion 均未开启。

报告文件保持 write-once 未改写；执行后仅修正 V11 executor 的复用式 report-assembly 元数据覆盖逻辑，供后续输出避免继承 V10 的描述性字段。该静态修正不改变本次科学 artifact、数值、gate 或 one-shot 边界，也不构成重跑或新的 scientific evidence。

## V10 low-rank OLS-to-generator remap canary（2026-09-12，INVALID；one-shot preflight stop）

按用户明确授权 `EXECUTE_LIGHTGEN_2D_LOW_RANK_OLS_TO_GENERATOR_REMAP_CANARY_V10`，在
`research/lightgen-ols-native-semantic-generation-v1`、HEAD=`a9cd1749adfa690e282fd52a6425564fc2cf102c` 上物化 V10 READY contract，但唯一一次 bounded preflight 在 scientific forward 前以 `INVALID_HARNESS` 停止；按授权 `no retry` 规则未重跑、未执行 300-step scientific training、未执行 DEV/cyclic scientific evaluation。

- [V10 contract](../outputs/20260912_v10/runs/lightgen_2d_low_rank_ols_to_generator_remap_canary_contract_v10_attempt_01/lightgen_2d_low_rank_ols_to_generator_remap_canary_contract.v10.json)，SHA-256=`5efeb9d4b8e5e1e29fee3e3c445a52a201fbfabdd7f696a183284c71832ca0f1`，READY/valid_completed；semantic authority=`DEFINE_LIGHTGEN_2D_LOW_RANK_OLS_TO_GENERATOR_REMAP_CANARY_CONTRACT_V10`，父 authority 为 V9。
- [preflight report](../outputs/20260912_v10/runs/lightgen_2d_low_rank_ols_to_generator_remap_canary_preflight_v10_attempt_01/lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_preflight_report.v1.json)，SHA-256=`c0d1cf193de37c7ed40e0d979769ae93dc3c08193d57e43955a1260fecb25f00`；[lineage](../outputs/20260912_v10/runs/lightgen_2d_low_rank_ols_to_generator_remap_canary_preflight_v10_attempt_01/lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_preflight_lineage.v1.json)，SHA-256=`fc3585925c0720ef532df8251c660ef6cb47b249360139020c464014b249458f`；classification=`INVALID_V10_PREFLIGHT_HARNESS`。

失败原因已定位为入口 harness：firewall probe 对已带 `adapter.` 前缀的参数名再次加前缀，错误报告 `V10 combined probe u_real gradient missing`；另修正 complex gradient norm helper 不得丢弃 imaginary 分量。修复仅为静态入口修正，未重跑 preflight，因此不能转化为 scientific validity。该 attempt 的事实边界为：branch/HEAD 未变、worktree status 未变、scientific forward/update=`0`、checkpoint scientific=`0`、protected/test/sealed=`0`、retry=`NO`。V10 Outer Gate `V10_LOW_RANK_OLS_TO_GENERATOR_REMAP_SUPPORTED` unresolved；`training_authorized=false`、`stage_b_authorized=false`、`generation_readiness=not_ready`、`formal_promotion=false`。

## V10 low-rank OLS-to-generator remap canary attempt-02（2026-09-12，INVALID；one-shot preflight stop）

按用户明确授权，在同一 READY V10 contract、branch=`research/lightgen-ols-native-semantic-generation-v1`、HEAD=`a9cd1749adfa690e282fd52a6425564fc2cf102c` 上执行修复后的 bounded preflight attempt-02。入口在 scientific forward 前的 identity probe 以 `INVALID_V10_PREFLIGHT_HARNESS` 停止；按 one-shot 规则不重试，也未进入 300-step scientific training、DEV 或 cyclic scientific evaluation。

- [preflight report](../outputs/20260912_v10/runs/lightgen_2d_low_rank_ols_to_generator_remap_canary_preflight_v10_attempt_02/lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_preflight_report.v1.json)，SHA-256=`cf50ae067c28de08b9bf27eb88294b34998842b84833eaa464eb27a7ab119d74`；[lineage](../outputs/20260912_v10/runs/lightgen_2d_low_rank_ols_to_generator_remap_canary_preflight_v10_attempt_02/lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_preflight_lineage.v1.json)，SHA-256=`b599330fe454ea31d466a9c657b205ac6b530eeb1ae67f57215c7ac23bb746e4`。
- 失败原因：`V10 step-0 adapter changed encoded coefficients`。即 rank-9 adapter 的 step-0 identity probe 未满足 frozen tolerance；因此本次不能证明 double-prefix 或 complex-gradient-norm 修复已通过，也不能产生 V10 scientific conclusion。
- attempt-02 事实边界：scientific forward=`0`、scientific updates=`0`、scientific checkpoint=`0`、protected/official-test/sealed=`0`、retry=`NO`；branch、HEAD 和 worktree status 均未改变。V10 Outer Gate `V10_LOW_RANK_OLS_TO_GENERATOR_REMAP_SUPPORTED` 仍 unresolved；`training_authorized=false`、`stage_b_authorized=false`、`generation_readiness=not_ready`、`formal_promotion=false`。

## V10 step-0 identity engineering stabilization（2026-09-12，BLOCKED；冻结合同代数不一致）

在不读取 scientific TRAIN/DEV、不创建 attempt-03、且不修改 attempt-01/02 的前提下，对实际生产 `LowRankAdapter` 做了只读 algebraic/synthetic identity audit。attempt-02 的 probe 比较的是同一批 `encoded_result.z_ols.coefficients` 与实际 `adapter(z)` 输出，形状和复数表示一致；失败不是 probe 把不同表示或不同 port-order 的张量混比。

冻结 V10 contract 同时要求：`U=s[e_0,...,e_8]`、`V=s[e_9,...,e_17]`，其中 `s=0.01`，并使用 `T=I+UV^H`。按此真实生产实现，`UV^H=s^2 Σ_k e_k e_{k+9}^H`，因此残差有 9 个非零项，每项绝对值为 `1e-4`，`||UV^H||_F=3e-4`；实测 `max_abs(T-I)=1e-4`，任意复数向量的 identity allclose 失败。也就是说，contract 所称的“orthogonal-support”行支持并不能使同列配对的 `UV^H` 为零；要使乘积为零，必须改变冻结初始化/参数化或允许一个 factor 的有效列支持为空，均超出本任务允许范围。

因此 exact step-0 identity 与当前冻结 rank-9、双因子非零、行支持初始化不能同时成立；identity tolerance 不得放宽，adapter、rank、objective、lambda 和 contract 均未修改。本工程稳定化任务返回 `BLOCKED`，V10 attempt-03 未执行，scientific updates=`0`，V10 Outer Gate 仍 unresolved。

## V10.1 low-rank OLS-to-generator remap canary（2026-09-12，VALID_COMPLETED；唯一一次）

V10.1 semantic authority `DEFINE_LIGHTGEN_2D_LOW_RANK_OLS_TO_GENERATOR_REMAP_CANARY_CONTRACT_V10_1` 仅 supersede V10 不可能的 step-0 初始化条款：保持 `T_adapt=I_100+UV^H`、rank/factor width=`9`，改为 `U0=0.01*I_9`（rows 0..8）、`V0=0`。V10 的光学/OLS 路径、V9 anchor+transfer objective、`lambda_transfer=28.48464149920509`、Adam、300-step budget、数据角色、Outer Gate 和 Stage-A firewall 均未改变。

- [V10.1 contract](../outputs/20260912_v10_1/runs/lightgen_2d_low_rank_ols_to_generator_remap_canary_contract_v10_1_attempt_01/lightgen_2d_low_rank_ols_to_generator_remap_canary_contract.v10_1.json)，SHA-256=`45210d92e00ba01255a4340dfc837c514c110640097aa20821d8d5b6e0efab3d`；supersedes 旧 V10 contract `5efeb9d4...`，范围仅为初始化条款。
- [preflight report](../outputs/20260912_v10_1/runs/lightgen_2d_low_rank_ols_to_generator_remap_canary_preflight_v10_1_attempt_01/lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_preflight_report.v1.json)，SHA-256=`e651e51a3d022b4ec041107e313a6e774a8740b7cd9dbd34f6a50c57ef670bf0`；[lineage](../outputs/20260912_v10_1/runs/lightgen_2d_low_rank_ols_to_generator_remap_canary_preflight_v10_1_attempt_01/lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_preflight_lineage.v1.json)，SHA-256=`1eb2508d6a8965d78ed4070b508718b2cdb69084bc3d5ff4bdeeec84cd3cf5d1`；`AUTHORIZABLE`，scientific=`0`。
- [scientific input](../outputs/20260912_v10_1/runs/lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_scientific_attempt_01/lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_scientific_input.v1.json)，SHA-256=`681fde951c77ef23e601be9be773826eed4412e7f968949240027fe76caeabe4`；[report](../outputs/20260912_v10_1/runs/lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_scientific_attempt_01/lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_scientific_report.v1.json)，SHA-256=`0eb96990db673c82526c2ea37bd15afadb909487eb0afe3ff7af84788b6b4212`；[lineage](../outputs/20260912_v10_1/runs/lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_scientific_attempt_01/lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_scientific_lineage.v1.json)，SHA-256=`4b835f4fd7cae36750a7e28ba324cb07453440d68ee12187982db85a0704766f`。

Preflight 证据：`||U0||F=0.03`、`||V0||F=0`、`||U0V0^H||F=0`、`max_abs(T-I)=0`；standalone 与 production synthetic identity 均 exact/allclose；`grad_U=0`（允许）、`grad_V=0.0010775054`（anchor），combined total adapter gradient=`0.0016925142`，firewall total=`0.0048664680`；一次 disposable step 后 `||T-I||F=0.0004217633`，rank=`9`；CUDA 3-step smoke、checkpoint/save/reload/resume、exact-core synthetic 均通过。设备为 `cuda:0`、RTX 3060 Laptop、PyTorch `2.5.1`、CUDA build `12.1`；仅 3,600 个 adapter real/imag 参数可训练，encoder/generator/OLS/W_OLS/Smooth 全部冻结。

Scientific result：`VALID_COMPLETED`、updates=`300`，唯一分类与 Outer Gate 均为 `V10_LOW_RANK_OLS_TO_GENERATOR_REMAP_SUPPORTED`。DEV Smooth macro recall `0.1036→0.3336`，`Delta_R=+0.2300`，paired bootstrap 95% CI=`[+0.22099,+0.23960]`；known cyclic transfer retained=`true`，10/10 template 与 Smooth pair 为正，raw validity=`true`。作为代价，DEV image loss `0.0670616→0.2914308`；template-anchor macro `0.1706→0.4916`，correct-template margin `-0.0081025→+0.0002165`，OLS margin mean 保持 `0.00208249`，均为描述性机制指标。

adapter residual：step-0 `Fro=0`、spectral=`0`、reported rank=`0`；step-300 `Fro=6.42947`、spectral=`6.06980`、`max_abs=0.241144`。前 9 个 singular values 为 `6.06980, 2.06025, 0.49754, 0.05884, 0.00796, 0.00292, 0.000335, 0.000247, 0.000132`；其余为约 `1e-7` 以下的 float32 identity-subtraction noise。故默认 `torch.linalg.matrix_rank` diagnostic 报告 step-300=`100`，但参数化的理论 residual rank 仍受 factor width=`9` 约束；该数值诊断不是额外 Outer Gate。

Stage-A 独立结果：DEV step-300 Smooth macro=`0.3336 < 0.937`，`STAGE_A_FIREWALL_FAILED`；因此 Stage B、scale/prior、generation、protected/test/sealed read、formal promotion 均未开放。V10.1 已执行且不得 retry、second arm、rank sweep 或调参。

## V9 joint semantic remapping warmup scientific execution（2026-09-12，VALID_COMPLETED；一次且仅一次）

按用户明确授权 `EXECUTE_LIGHTGEN_2D_JOINT_SEMANTIC_REMAPPING_WARMUP_V9`，在
`research/lightgen-ols-native-semantic-generation-v1`、start/end HEAD=`a9cd1749adfa690e282fd52a6425564fc2cf102c` 上完成 V9 READY contract、bounded engineering preflight、exact-core synthetic 与唯一一次 300-step CUDA scientific arm。工作区既有 dirty 状态保持；branch、HEAD、status line 与科学运行前一致。preflight=`AUTHORIZABLE`、`scientific_300_step_executed=false`；scientific=`VALID_COMPLETED`，唯一分类=`V9_JOINT_SEMANTIC_REMAPPING_NOT_SUPPORTED`。

- [V9 contract](../outputs/20260912_v9/runs/lightgen_2d_joint_semantic_remapping_warmup_contract_v9_attempt_01/lightgen_2d_joint_semantic_remapping_warmup_contract.v9.json)，SHA-256=`4d0fb81bfa6152995aad4484c83487c20335616283097b394757f4616bf41fca`，READY/valid_completed；semantic authority=`DEFINE_LIGHTGEN_2D_JOINT_SEMANTIC_REMAPPING_WARMUP_CONTRACT_V9`；父 authority 为 V8 contract。
- preflight [report](../outputs/20260912_v9/runs/lightgen_2d_joint_semantic_remapping_warmup_preflight_v9_attempt_01/lightgen_2d_joint_semantic_remapping_warmup_v9_preflight_report.v1.json)，SHA-256=`8baa04ac089ff82aebf3fefb9de9ff70a0d495022b5c19e852ab8eb82926e2f8`；[lineage](../outputs/20260912_v9/runs/lightgen_2d_joint_semantic_remapping_warmup_preflight_v9_attempt_01/lightgen_2d_joint_semantic_remapping_warmup_v9_preflight_lineage.v1.json)，SHA-256=`002fd2d5303682f30f6b0b54f580acd99f6bd8456d2d55c441215855e0e7fb7c`；classification=`LIGHTGEN_2D_JOINT_SEMANTIC_REMAPPING_WARMUP_V9_AUTHORIZABLE`。
- [scientific input](../outputs/20260912_v9/runs/lightgen_2d_joint_semantic_remapping_warmup_v9_scientific_attempt_01/lightgen_2d_joint_semantic_remapping_warmup_v9_scientific_input.v1.json)，SHA-256=`6cfedfb607834bd99784f7082535628c727436c9006216e67781afbe326b4345`；[report](../outputs/20260912_v9/runs/lightgen_2d_joint_semantic_remapping_warmup_v9_scientific_attempt_01/lightgen_2d_joint_semantic_remapping_warmup_v9_scientific_report.v1.json)，SHA-256=`280cd33abbe3195856798989d7f300c9fcc618e57906149c3abc0abdfd3d7465`；[lineage](../outputs/20260912_v9/runs/lightgen_2d_joint_semantic_remapping_warmup_v9_scientific_attempt_01/lightgen_2d_joint_semantic_remapping_warmup_v9_scientific_lineage.v1.json)，SHA-256=`47d8768165b5d5207df8a190547b56223b49f4a9cf89e7da9158be43f05fa418`。
- [sufficient observations](../outputs/20260912_v9/runs/lightgen_2d_joint_semantic_remapping_warmup_v9_scientific_attempt_01/lightgen_2d_joint_semantic_remapping_warmup_v9_sufficient_observations.v1.pt)，SHA-256=`48cff565de6144d941715dd497c47c6fccc956e6e0117ad896b49e76b82f33ff`；[step-300 cyclic raw outputs](../outputs/20260912_v9/runs/lightgen_2d_joint_semantic_remapping_warmup_v9_scientific_attempt_01/lightgen_2d_joint_semantic_remapping_warmup_v9_step300_cyclic_raw_outputs.v1.pt)，SHA-256=`8556a1e8db945d73cf4dab680348b5b7eec8af737655d5561828a7a416c1784f`；sufficient/raw/report/lineage reload=`PASS`。

冻结 protocol：exact V3 CONTROL step-300 encoder 与 exact V4 valid step-300 generator 作为 step-0；仅 `encoder.phase` 与 `generator.phase` 可训练（各 `20,000` 个参数，合计 `40,000`）。100-port complex OLS、modal projection/re-emission、`T=I_100`、shared `W_OLS`、V5 TRAIN-only prototypes、历史 TRAIN templates、SmoothMNISTEvaluator v2.1 与其余 runtime state 全部冻结。目标保持 V8 不变：`L_V9=L_anchor+lambda_transfer*L_transfer`；`s_tau=0.02919053753794475`，复用固定 `lambda_transfer=28.48464149920509`，未重算、未裁剪、未调参。Adam `lr=1e-3`、batch=`10`、grid=`2048`、300 updates、checkpoint/evaluation=`0/100/200/300`；TRAIN=`55,000`，DEV=`5,000`，cyclic selection=`1,000/100`，seed=`4321`，paired bootstrap=`B=999, seed=271828`。

运行设备为 `cuda:0`、NVIDIA GeForce RTX 3060 Laptop GPU、compute capability=`8.6`、PyTorch=`2.5.1`、CUDA=`12.1`。preflight 的 anchor/transfer 梯度对 encoder 分别为 `0.0082287618885706`/`0.00028888416448265975`，对 generator 分别为 `0.009280449860086344`/`0.00019965410761688384`；joint trainable firewall、negative controls、image/prototype/Smooth gradient exclusion、3-step CUDA update、checkpoint/reload/RNG/sampler resume 与 exact-core synthetic 均通过。

| step | TRAIN L_anchor | TRAIN L_transfer | TRAIN L_total | TRAIN L_image(eval-only) | DEV L_image | Smooth macro | template-anchor macro | correct-template margin | OLS margin mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.22846563 | 0.69342184 | 21.98033819 | 0.07510780 | 0.06706161 | 0.1036 | 0.1706 | -0.00810254 | 0.00208249 |
| 100 | 2.20848503 | 0.69193470 | 21.91799689 | 0.06499042 | 0.06625223 | 0.1054 | 0.2050 | -0.00670630 | 0.00206976 |
| 200 | 2.20736347 | 0.68992402 | 21.85960196 | 0.06401648 | 0.06622797 | 0.1100 | 0.2386 | -0.00568535 | 0.00204236 |
| 300 | 2.16701212 | 0.68808337 | 21.76682022 | 0.06836181 | 0.06666521 | 0.1184 | 0.2670 | -0.00486589 | 0.00200970 |

step-300 相对 step-0：natural Smooth `Delta_R=+0.014800`，paired bootstrap 95% CI=`[+0.011400,+0.018610]`，CI lower 为正但未达到预注册 `Delta_R>=+0.05`，故 natural Smooth criterion=`FAIL`；DEV image Delta=`-0.0003963965`；template-anchor macro change=`+0.0964`，correct-template margin change=`+0.00323665`，均为 descriptive mechanism evidence；OLS margin Delta=`-0.0000727933`，仅 descriptive。

step-300 known cyclic transfer retained=`true`：template observed mean/median=`0.01088561/0.00826996`，95% CI=`[0.01061948,0.01115307]`，p_positive=`1.0`；Smooth observed mean/median=`3.10276725/2.93491745`，95% CI=`[2.99842256,3.20315371]`，p_positive=`1.0`；template/Smooth positive pair counts=`10/10`，`broad=true`，`pair_dominated=false`。

| cyclic pair | template Delta mean | Smooth Delta mean |
| --- | ---: | ---: |
| 0 -> 1 | 0.02981124 | 4.79306646 |
| 1 -> 2 | 0.01347965 | 4.17880627 |
| 2 -> 3 | 0.00287232 | 0.27432058 |
| 3 -> 4 | 0.01103936 | 5.93916874 |
| 4 -> 5 | 0.00493736 | 2.87124777 |
| 5 -> 6 | 0.00494553 | 1.57296430 |
| 6 -> 7 | 0.00854289 | 3.56916991 |
| 7 -> 8 | 0.00844004 | 2.67373318 |
| 8 -> 9 | 0.00521505 | 1.36521842 |
| 9 -> 0 | 0.01957267 | 3.78997682 |

step 0/100/200/300 DEV 与 cyclic base/shift raw 均 finite、shape-valid、structured、nonconstant；step-0/300 exact duplicate output=`0`，unique=`5000`，near-identical threshold 未宣称支持。Primary Gate 三项中 `known_cyclic_transfer_retained=PASS`、`raw_validity=PASS`、`natural_smooth_improvement=FAIL`，所以 V9 Outer Gate `V9_JOINT_SEMANTIC_REMAPPING_SUPPORTED` 未满足。该结果支持“joint objective 产生小幅独立 Smooth 正增益且保留已知 cyclic transfer”的 bounded fact，但不支持达到 V9 semantic remapping 门槛，也不能单独证明 generator adaptability assumption。

Stage-A firewall 独立保持 `R300>=0.937`；本次 `R300=0.1184`，状态=`STAGE_A_FIREWALL_FAILED`。确认：`encoder update=YES`、`generator update=YES`、`L_image gradient=NO`、`prototype gradient=NO`、`Smooth/Frozen gradient=NO`；`retry=NO`、`Stage B=NO`、`generation=NO`、`protected/test/sealed=NO`、`formal promotion=NO`。`training_authorized=false`、`stage_b_authorized=false`、`generation_readiness=not_ready`、`formal_promotion=false` 保持。

## V8 absolute-anchor plus transfer warmup scientific execution（2026-09-12，VALID_COMPLETED；一次且仅一次）

按用户明确授权 `EXECUTE_LIGHTGEN_2D_ABSOLUTE_ANCHOR_PLUS_TRANSFER_WARMUP_V8`，在
`research/lightgen-ols-native-semantic-generation-v1`、start/end HEAD=`a9cd1749adfa690e282fd52a6425564fc2cf102c` 上完成 V8 READY contract、一次 bounded engineering preflight 与唯一一次 300-step CUDA scientific arm。工作区原有 dirty 状态保持不变；branch、HEAD 与 status line 未改变。preflight 为 `AUTHORIZABLE`，其 `scientific_300_step_executed=false`；scientific 结果为 `VALID_COMPLETED`，唯一分类为 `V8_ABSOLUTE_ANCHOR_PLUS_TRANSFER_NOT_SUPPORTED`。

- [V8 contract](../outputs/20260911_v8/runs/lightgen_2d_absolute_anchor_plus_transfer_warmup_contract_v8_attempt_01/lightgen_2d_absolute_anchor_plus_transfer_warmup_contract.v8.json)，SHA-256=`02cef2918ff89eafaa53cf52f7a7492b3091e84acba3c351b6142b94b703a85d`，READY/valid_completed；semantic authority=`DEFINE_LIGHTGEN_2D_ABSOLUTE_ANCHOR_PLUS_TRANSFER_WARMUP_CONTRACT_V8`。
- preflight [report](../outputs/20260911_v8/runs/lightgen_2d_absolute_anchor_plus_transfer_warmup_preflight_v8_attempt_01/lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_preflight_report.v1.json)，SHA-256=`e197bff80ea4fbb42d2a0c141028ecf7cabe34416d5816df6a95620cc9a3faad`；[lineage](../outputs/20260911_v8/runs/lightgen_2d_absolute_anchor_plus_transfer_warmup_preflight_v8_attempt_01/lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_preflight_lineage.v1.json)，SHA-256=`c2dcd73581143d288b5e536252f7188cfeabb82c624d6756444ec868fbf9b2c2`；preflight classification=`LIGHTGEN_2D_ABSOLUTE_ANCHOR_PLUS_TRANSFER_WARMUP_V8_AUTHORIZABLE`。
- [scientific input](../outputs/20260911_v8/runs/lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_scientific_attempt_01/lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_scientific_input.v1.json)，SHA-256=`4942c79d1ed4df92ddbee5cfbcb9771e749750c3ceeaffac652869c2e4b8673d`；[report](../outputs/20260911_v8/runs/lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_scientific_attempt_01/lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_scientific_report.v1.json)，SHA-256=`5d534f9f7e7192c807a3c927a0815680c17b5d00d7f9edaf8c9987cf3bd39806`；[lineage](../outputs/20260911_v8/runs/lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_scientific_attempt_01/lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_scientific_lineage.v1.json)，SHA-256=`69cfcf2e34e02d192c1f7cbca5f98bc960c163371a28404d3fc0fff9a0fad121`。
- [sufficient observations](../outputs/20260911_v8/runs/lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_scientific_attempt_01/lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_sufficient_observations.v1.pt)，SHA-256=`dd3848bdcf85c85d06eb0ca220b9f4149ff8df52701989416aa563bb6441a415`；[step-300 cyclic raw outputs](../outputs/20260911_v8/runs/lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_scientific_attempt_01/lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_step300_cyclic_raw_outputs.v1.pt)，SHA-256=`9599e0f52bfef49cbf74c615a39be12460454cf16c65e4a1c115049fe4814ce5`；sufficient/raw/report/lineage reload 均通过。

冻结 protocol：exact V3 CONTROL step-300 encoder 初值；exact V4 step-300 generator；100-port complex OLS、modal projection/re-emission、`T=I_100`、`W_OLS`、V5 TRAIN-only prototypes、历史 TRAIN templates 与 SmoothMNISTEvaluator v2.1 全部冻结；仅 `encoder.phase` 可训练。Adam `lr=1e-3`、batch=`10`、grid=`2048`、300 updates、checkpoint/evaluation=`0/100/200/300`。目标为
`L_V8=L_anchor+lambda_transfer*L_transfer`：自然 raw `r=G(E(x))` 的 absolute anchor 使用
`b_k=-d_image(r,tau_k)/(s_tau+1e-12)` 与 cross-entropy，transfer 使用
`z_shift=z-mu_y+mu_t`、`t=(y+1) mod 10`、`mean(softplus(-Delta_template))`；image MSE、prototype CE 与 Smooth evaluator 无梯度，仅作 evaluation/shift support。TRAIN=`55,000`，DEV=`5,000`，cyclic selection=`1,000/100`，seed=`4321`，paired bootstrap=`B=999, seed=271828`。

冻结的 absolute-anchor 常数为 `s_tau=0.02919053753794475`（45 个 unordered template pairs，min/median/max=`0.0079232950/0.0291905375/0.0729670815`）。首个确定性 TRAIN batch 的初始梯度为 `g_anchor=0.0082287618885706`、`g_transfer=0.00028888416448265975`，按唯一规则
`lambda_transfer=g_anchor/(g_transfer+1e-12)=28.48464149920509`，未裁剪、未调参、未重算。preflight 通过 encoder-only、generator/OLS/Smooth firewall、finite/nonzero gradients、negative controls、production no-data smoke、真实 3-step update 与 checkpoint/reload/resume、exact-core synthetic；synthetic 无科学含义。

运行设备为 `cuda:0`、NVIDIA GeForce RTX 3060 Laptop GPU、compute capability=`8.6`、VRAM=`6441926656` bytes、PyTorch=`2.5.1`、CUDA=`12.1`。

| step | TRAIN L_anchor | TRAIN L_transfer | TRAIN L_total | TRAIN L_image(eval-only) | DEV L_image | Smooth macro | OLS margin mean | template-anchor macro |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.22846563 | 0.69342184 | 21.98033819 | 0.07510780 | 0.06706161 | 0.1036 | 0.00208249 | 0.1706 |
| 100 | 2.23859913 | 0.69290392 | 21.97571897 | 0.06450775 | 0.06671077 | 0.1040 | 0.00206943 | 0.1826 |
| 200 | 2.24820785 | 0.69156806 | 21.94727610 | 0.06408656 | 0.06650268 | 0.1062 | 0.00203811 | 0.1914 |
| 300 | 2.22151052 | 0.69053010 | 21.89101275 | 0.06882872 | 0.06643724 | 0.1086 | 0.00199363 | 0.2022 |

step-300 相对 step-0：natural Smooth `Delta_R=+0.005000`，paired bootstrap 95% CI=`[+0.002800,+0.007400]`，但预注册要求为 `Delta_R>=+0.05` 且 CI lower `>0`，故 natural gate=false；DEV image Delta=`-0.00062436697`；OLS margin mean Delta=`-0.00008886290`。template-anchor 仅作描述性 objective diagnostic：macro recall=`0.1706→0.2022`，correct-template margin mean=`-0.00810254→-0.00665125`，仍为负，未被作为额外 post-hoc Gate。

cyclic transfer retained=true：template observed mean=`0.00583127524`，95% CI=`[0.00562985606,0.00603236101]`；Smooth observed mean=`1.94132099`，95% CI=`[1.83938933,2.04307683]`；template/Smooth positive pair counts=`10/10`，`broad=true`，`pair_dominated=false`。step 0/300 collapse diagnostics 已写入：overall raw std=`27.7540→26.9755`，between-class mean-output separation=`0.00334851→0.00334354`，exact duplicate outputs=`0` 且 unique=`5000`；near-identical threshold primitive 未宣称支持。所有 step-0/100/200/300 DEV 与 cyclic base/shift raw 均 finite、shape-valid、structured、nonconstant。

因此 V8 Outer Gate `V8_ABSOLUTE_ANCHOR_PLUS_TRANSFER_SUPPORTED` 未满足；本次只得到 bounded objective/transfer evidence，不能解释为 Stage-A semantic eligibility。Stage-A firewall 独立保持 `R300>=0.937`，本次 `R300=0.1086`，状态=`STAGE_A_FIREWALL_FAILED`。没有 retry、second arm、hyperparameter tuning、architecture/OLS/W_OLS/generator change、scale/prior、generation、Stage B、protected/test/sealed read 或 formal promotion；`training_authorized=false`、`stage_b_authorized=false`、`generation_readiness=not_ready`、`formal_promotion=false` 保持。

## V7 transfer-aware semantic warmup scientific execution（2026-09-11，VALID_COMPLETED；一次且仅一次）

按明确授权 EXECUTE_LIGHTGEN_2D_TRANSFER_AWARE_SEMANTIC_WARMUP_V7，在 research/lightgen-ols-native-semantic-generation-v1、start/end HEAD=a9cd1749adfa690e282fd52a6425564fc2cf102c 上完成 V7 contract、bounded preflight 与唯一一次 300-step scientific arm。preflight attempt-01/02 均在 scientific forward 前因可定位的 synthetic harness 错误停止并保持 immutable；attempt-03 通过后未重跑 scientific。工作区原有 dirty 状态保持不变，branch/HEAD 未改变。

- [V7 contract](../outputs/20260911_v7/runs/lightgen_2d_transfer_aware_semantic_warmup_contract_v7_attempt_01/lightgen_2d_transfer_aware_semantic_warmup_contract.v7.json)，SHA-256=974d6dc560d2f4427cc75e5dc3d3404140d6e2a84439d800f6af10e299bc7766，结果=READY/valid_completed。
- preflight attempt-03：[report](../outputs/20260911_v7/runs/lightgen_2d_transfer_aware_semantic_warmup_preflight_v7_attempt_03/lightgen_2d_transfer_aware_semantic_warmup_v7_preflight_report.v1.json)，SHA-256=18ea57a4bd2baa5b9a975a5902f899e9bb8d377964ecbfde9dd01b3165aa332e；[lineage](../outputs/20260911_v7/runs/lightgen_2d_transfer_aware_semantic_warmup_preflight_v7_attempt_03/lightgen_2d_transfer_aware_semantic_warmup_v7_preflight_lineage.v1.json)，SHA-256=fb1644c17fd1c6ae308ab5e9e5822e5196a165098f91ff34d8bc6cfe456da79f；AUTHORIZABLE，scientific_300_step_executed=false。
- scientific input SHA-256=2f1ff0a200d6e7ef42859276ec2727aaf2ce36fe73d16f5fa6d4f8fa3d25ceb3；[report](../outputs/20260911_v7/runs/lightgen_2d_transfer_aware_semantic_warmup_v7_scientific_attempt_01/lightgen_2d_transfer_aware_semantic_warmup_v7_scientific_report.v1.json)，SHA-256=ca06e5efe91c51f3b2ebff5125986e6599e607100f9c5932c4d504d55a2482fc；[lineage](../outputs/20260911_v7/runs/lightgen_2d_transfer_aware_semantic_warmup_v7_scientific_attempt_01/lightgen_2d_transfer_aware_semantic_warmup_v7_scientific_lineage.v1.json)，SHA-256=4bac18f7b0bb588bb29cc75ab2087193ab2c44b26a7dc4e73f605085fe300139。

训练固定为 exact V3 CONTROL step-300 encoder 初值、exact V4 step-300 generator 全冻结、V5 TRAIN-only prototypes、历史 TRAIN templates、T=I_100 与 W_OLS；仅 encoder.phase 可训练。Adam lr=1e-3、batch=10、300 updates、grid=2048；唯一目标为 L_V7=L_transfer=mean(softplus(-Delta_template))，其中 z_shift=z-mu_y+mu_t、t=(y+1) mod 10；image loss、prototype loss、Smooth/Frozen evaluator 均无梯度且仅作评估或描述。DEV=5000 rows，step=0/100/200/300；cyclic=1000/100，seed=4321。

运行设备：cuda:0，NVIDIA GeForce RTX 3060 Laptop GPU，compute capability=8.6，VRAM=6441926656 bytes，PyTorch=2.5.1，CUDA build=12.1。

| step | TRAIN L_transfer | TRAIN L_image | DEV L_image | Smooth macro recall | OLS margin mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.69342184 | 0.07510780 | 0.06706161 | 0.1036 | 0.00208249 |
| 100 | 0.69244408 | 0.06457431 | 0.06692999 | 0.1036 | 0.00207652 |
| 200 | 0.69080558 | 0.06446711 | 0.06679273 | 0.1036 | 0.00205580 |
| 300 | 0.68953269 | 0.06921547 | 0.06668990 | 0.1042 | 0.00202175 |

step-300 相对 step-0：Smooth observed Delta_R=+0.000600，95% CI=[-0.000600,+0.001800]，未达到预注册 +0.05 且 CI lower 不大于 0；DEV image Delta=-0.000371713；TRAIN transfer-loss Delta=-0.00388916；OLS prototype-margin Delta=-6.07760e-5，95% CI=[-6.43653e-5,-5.71490e-5]，仅 secondary descriptive。

cyclic step-300 transfer 成功保留：template observed mean/median=0.00816400/0.00616277，95% CI=[0.00792898,0.00840002]，p_positive=1.0；Smooth observed mean/median=2.53695471/2.49440956，95% CI=[2.43122264,2.64568621]，p_positive=1.0；template/Smooth positive pair counts=10/10，broad=true，pair_dominated=false，known_cyclic_transfer_retained=true。

| cyclic pair | template Delta mean | Smooth Delta mean |
| --- | ---: | ---: |
| 0 -> 1 | 0.02126103 | 2.63885640 |
| 1 -> 2 | 0.01058828 | 3.32051782 |
| 2 -> 3 | 0.00196776 | 0.22176126 |
| 3 -> 4 | 0.00748423 | 4.37591817 |
| 4 -> 5 | 0.00402526 | 2.81977525 |
| 5 -> 6 | 0.00586092 | 2.10184672 |
| 6 -> 7 | 0.00844589 | 3.64366655 |
| 7 -> 8 | 0.00417849 | 1.49608160 |
| 8 -> 9 | 0.00312327 | 1.42380861 |
| 9 -> 0 | 0.01470491 | 3.32731475 |

step 0/100/200/300 DEV raw 与 cyclic base/shift raw 均 finite、shape-valid、structured、nonconstant。Primary Gate 要求 natural Smooth improvement、known cyclic transfer 与 raw validity 全部通过；本次仅后两者通过，因此唯一分类为 V7_TRANSFER_AWARE_SEMANTIC_WARMUP_NOT_SUPPORTED，Outer Gate V7_TRANSFER_AWARE_SEMANTIC_WARMUP_SUPPORTED 未满足。

Stage-A firewall 独立为 R300=0.1042 < 0.937，状态=STAGE_A_FIREWALL_FAILED；Stage B、generation、formal promotion 均关闭。没有 retry、second arm、lr/loss tuning、target/magnitude sweep、generator/architecture/OLS/W_OLS/optical update、scale、prior、protected/test/sealed read 或 post-hoc Gate change。

## V6 proto-only semantic warmup scientific execution（2026-09-11，VALID_COMPLETED；一次且仅一次）

按明确授权 EXECUTE_LIGHTGEN_2D_PROTO_ONLY_SEMANTIC_WARMUP_V6，在 research/lightgen-ols-native-semantic-generation-v1、start/end HEAD=a9cd1749adfa690e282fd52a6425564fc2cf102c 上先完成 bounded preflight，随后执行唯一一次 300-step scientific arm。preflight attempt-01/02 均在 scientific forward 前因 harness 问题停止并保持 immutable；attempt-03 通过后未重跑 scientific arm。工作区原有 dirty 状态保持不变，branch/HEAD 未改变。

- [V6 contract](../outputs/20260911_v6/runs/lightgen_2d_proto_only_semantic_warmup_contract_v6_attempt_01/lightgen_2d_proto_only_semantic_warmup_contract.v6.json)，SHA-256=4b849632b67bd68f4075816e061bb3db0664abe31ab757a7137bfcd4c8f9f441，结果=READY/valid_completed。
- preflight attempt-03：[report](../outputs/20260911_v6/runs/lightgen_2d_proto_only_semantic_warmup_preflight_v6_attempt_03/lightgen_2d_proto_only_semantic_warmup_v6_preflight_report.v1.json)，SHA-256=d45082a1384f1001a63b69972203bf99d4d9095243c6ef5217f6d68a258559be；[lineage](../outputs/20260911_v6/runs/lightgen_2d_proto_only_semantic_warmup_preflight_v6_attempt_03/lightgen_2d_proto_only_semantic_warmup_v6_preflight_lineage.v1.json)，SHA-256=15fef1def813b9ea6fd3f5dd33bf65c76e56fd7a782dc67b16a629474050054e；AUTHORIZABLE，scientific_300_step_executed=false。
- scientific input SHA-256=c27d02af5acfd684136a01e857123c2d58872072ea0bb27cb238fb3be57b9550；[report](../outputs/20260911_v6/runs/lightgen_2d_proto_only_semantic_warmup_v6_scientific_attempt_01/lightgen_2d_proto_only_semantic_warmup_v6_scientific_report.v1.json)，SHA-256=67918a6fe1dc4200b389a0ca2c4b419a6c150fb2f11288c1adc00e72605d09e9；[lineage](../outputs/20260911_v6/runs/lightgen_2d_proto_only_semantic_warmup_v6_scientific_attempt_01/lightgen_2d_proto_only_semantic_warmup_v6_scientific_lineage.v1.json)，SHA-256=40153c45103e870c9ee3188feab18e141bfae31f18b98c611f21b6e8a5be1ac1。

训练严格固定为 V3 CONTROL step-300 encoder 初值、V4 valid step-300 generator 初值、V5 TRAIN-only prototypes 与 s_mu=0.00555040966719389；仅更新 encoder.phase，generator.phase、OLS、Smooth evaluator 与 prototypes 冻结；Adam lr=1e-3、batch=10、300 updates；优化目标为 L_V6=L_proto，image MSE 仅 evaluation-only。DEV 为完整 5000 rows，在 step 0/100/200/300 读取；step-300 cyclic 使用冻结 1000/100 selection、seed=4321、t=(y+1) mod 10。

| step | evaluation L_image | Smooth macro recall | OLS margin mean |
| ---: | ---: | ---: | ---: |
| 0 | 0.06706161 | 0.1036 | 0.00208249 |
| 100 | 0.06742571 | 0.1030 | 0.00211740 |
| 200 | 0.06777967 | 0.1030 | 0.00214080 |
| 300 | 0.06808599 | 0.1030 | 0.00215925 |

step-300 相对 step-0：Delta_L_image=+0.0010243780；OLS margin Delta=+7.67765e-5，paired bootstrap 95% CI=[+7.32591e-5,+8.00236e-5]，p_positive=1.0；Smooth recall Delta=-0.0005960，95% CI=[-0.0014,0.0]，p_positive=0.0。step-300 Smooth per-class recall=[0,0,0.004,0,0.026,0,0.002,0,0.998,0]。

| step-300 cyclic metric | observed mean [95% bootstrap CI] | pass |
| --- | ---: | --- |
| template transfer Delta | -0.00172974 [-0.00188013,-0.00157520] | false |
| Smooth transfer Delta | -0.17172088 [-0.26391294,-0.07869869] | false |

所有 DEV/cyclic raw 输出均 finite、shape-valid、structured、nonconstant；但 cyclic positive pairs 仅 template 1/10、Smooth 4/10，broad=false、pair_dominated=true、known_cyclic_transfer_retained=false。因此唯一分类为 V6_PROTO_ONLY_SEMANTIC_WARMUP_NOT_SUPPORTED：本次 bounded evidence 只支持 OLS margin mechanism pass，不支持 semantic warmup gate。

Outer Gate V6_PROTO_ONLY_SEMANTIC_WARMUP_SUPPORTED 未满足。Stage-A firewall 独立保持 R300=0.103，要求仍为 R300>=0.937，状态=STAGE_A_FIREWALL_FAILED；Stage B 未授权。没有 scientific retry、第二 hyperparameter arm、scale extraction、prior、generation、architecture/OLS/W_OLS/optical update、protected/test/sealed read 或 formal promotion。

## V5 class-direction transfer operating-point stability diagnosis（2026-09-11，VALID_COMPLETED；一次且仅一次）

按明确授权 `DIAGNOSE_V5_CLASS_DIRECTION_TRANSFER_OPERATING_POINT_STABILITY_V1`，在 `research/lightgen-ols-native-semantic-generation-v1`、start/end HEAD=`a9cd1749adfa690e282fd52a6425564fc2cf102c` 上完成一次 forward-only/no-update diagnosis。历史有效 authority `OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_SUPPORTED`、冻结 V5 CONTROL/CANDIDATE step-300 checkpoints、V5 选择索引与固定 cyclic rule `t=(y+1) mod 10` 均通过验证；未执行 optimizer、backward、parameter/optical update、V5 retry、Stage B、generation 或 protected/test/sealed read。

结果=`VALID_COMPLETED/valid_completed`，唯一分类=`V5_CONTROL_OPERATING_POINT_TRANSFER_LOSS_SUPPORTED`；Outer Gate `V5_CLASS_DIRECTION_TRANSFER_OPERATING_POINT_STABILITY_RESOLVED` 已满足。该结果只证明本次固定操作点的 bounded transfer-loss stability，不改变 V5 latent-to-optical transfer-coupling 的 unresolved 状态，也不构成训练、Stage B、generation 或 formal promotion authorization。

- [`input`](../outputs/20260911_v5/runs/lightgen_2d_v5_class_direction_transfer_operating_point_stability_v1_attempt_01/lightgen_2d_v5_class_direction_transfer_operating_point_stability_input.v1.json)，SHA-256=`2d4dce07d24b8e6b32a37e3570c6ec1bdc6ab1b2f18dc11999cf434ef7e85aed`；[`report`](../outputs/20260911_v5/runs/lightgen_2d_v5_class_direction_transfer_operating_point_stability_v1_attempt_01/lightgen_2d_v5_class_direction_transfer_operating_point_stability_report.v1.json)，SHA-256=`1d8e0e232d9a5ef05fa964688c11847d960466d6b7e50cef71d7162b7d953ee`；
- [`lineage`](../outputs/20260911_v5/runs/lightgen_2d_v5_class_direction_transfer_operating_point_stability_v1_attempt_01/lightgen_2d_v5_class_direction_transfer_operating_point_stability_lineage.v1.json)，SHA-256=`67f1b77594e944b5313b82f72574dde7ff21d2da54c3bcd97efd7386f34fea20`；sufficient observations SHA-256=`7d0281d66426906a648fc8d9aa284e563b6492cdd3011b06ddfb68db84c93a25`；attempt-03 audit artifact SHA-256=`c137313ecc5c3a0f8056a5be213bf5d4b1e1565316ecb5eeeb85aa5f7f2f0612`，audit=`PASS`。
- scientific forward count=`600`（CONTROL/CANDIDATE encoder 各 100；四个 optical endpoints 各 100），Smooth evaluations=`400`；selection=`1000/100`，seed=`4321`；paired bootstrap=`B=999, seed=271828`，raw/sufficient/bootstrap reload 均为 `PASS`。

| operating point | template Δ mean [95% CI] | Smooth Δ mean [95% CI] | template transfer | Smooth transfer |
| --- | ---: | ---: | --- | --- |
| CONTROL | `-0.00099143 [-0.00113745, -0.00084232]` | `-0.05296442 [-0.15229365, 0.03803239]` | `false` | `false` |
| CANDIDATE | `-0.00127322 [-0.00142480, -0.00111903]` | `-0.09730077 [-0.19547855, -0.00736787]` | `false` | `false` |

两操作点的 raw endpoint 均 finite、shape-valid、structured、nonconstant；正向 pair count 均为 template 3/10、Smooth 3/10，descriptive direction 为 broad，但 primary transfer flags 均为 `false`。因此本次外门分类依赖固定模板方向的操作点 transfer-loss stability，不能外推为全局生成能力或下一阶段授权。

## V5 latent-to-optical transfer-coupling diagnosis attempt-03（2026-09-11，VALID_COMPLETED）

一次且仅一次的 forward-only/no-update scientific diagnosis 已完成。branch=`research/lightgen-ols-native-semantic-generation-v1`，start/end HEAD=`a9cd1749adfa690e282fd52a6425564fc2cf102c`；attempt-03 入口消费冻结 producer authority，并按授权允许 producer HEAD 与当前 consumer HEAD 不同。

结果=`VALID_COMPLETED/valid_completed`，唯一分类=`V5_LATENT_TO_OPTICAL_TRANSFER_MIXED_OR_UNRESOLVED`；Outer Gate `V5_LATENT_TO_OPTICAL_TRANSFER_MECHANISM_RESOLVED` 仍 unresolved。attempt-01/02 INVALID 历史保持 immutable。

- [`attempt-03 report`](../outputs/20260911_v5/runs/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_v1_attempt_03/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_report.v1.json)，SHA-256=`b759edd433cf53961d8b1a4f9cf61a341fd5a75a56c0e2985969bcf1799f791e`；[`lineage`](../outputs/20260911_v5/runs/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_v1_attempt_03/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_lineage.v1.json)，SHA-256=`71c11240c87b79478c047fd9d6688f0e2f5a789e3d76a2e616bccd72d22eb6bc`；
- sufficient observations SHA-256=`8985d1b48cb96e1f684ee263ebc050144802cf610c9806bed9cd21fc1c49b6ef`，reload=`PASS`；selection=`1000/100`，`SELECTION_SEED=4321` 重建并匹配 frozen index SHA=`4acbae34a68778753d770f2cac955a8e109e438fb4df9f41e83b15df46890bb8`；DEV reads=`1000`，scientific forward=`600`，Smooth evaluations=`400`。
- `||delta_V||` mean/median=`8.618106e-6/8.632646e-6`；`||delta_class||`=`6.110225e-5/6.064601e-5`；ratio mean/median=`0.152174/0.151641`；rho mean/median=`0.228568/0.216745`；alignment mean/median=`0.089892/0.108981`。
- Smooth ΔS actual/match/full 的 mean=`-0.140839/-0.070122/-0.502025`，paired 95% CI 分别为 `[-0.154794,-0.125540]`、`[-0.084865,-0.056285]`、`[-0.620633,-0.398167]`；三个 primary flags 全为 `false`。四个 raw endpoint 均 finite、shape-valid、structured、nonconstant；方向一致性 descriptive result=`broad`，但连续 Smooth 证据为负。

因此该结果不支持 subthreshold、latent-direction misalignment 或 magnitude-limited 三个排他分类，按预注册规则返回 mixed/unresolved。optimizer/backward/update/training/V5 retry/V6/Stage B/generation/protected-test-sealed/formal promotion 全部为 NO。

## V5 transfer-coupling production entry stabilization（2026-09-11，VALID_COMPLETED；非 scientific execution）

本地 bounded engineering stabilization 已完成。起始 branch=`research/lightgen-ols-native-semantic-generation-v1`、HEAD=`9a72e0bcf2de12ab9a0caa7151f273de76768dc7`；focused commit=`a9cd1749adfa690e282fd52a6425564fc2cf102c`；当前 HEAD=`a9cd1749adfa690e282fd52a6425564fc2cf102c`；入口 SHA-256=`ca7d0a987f6e2f01d6819511b4124980b91fa7fb02885e2a013ba31682f73c02`。

- [`stabilized entry`](../scripts/research/diagnose_lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_v1.py) 的真实 authority no-data dry-run 通过：accepted preflight 使用真实嵌套 `accepted_preflight.input.sha256` schema；V5 scientific attempt-01、注册 class-direction selection、V4 generator、V5 control/candidate、Smooth v2.1 与 V2 runtime 解析及 loadability/schema 校验通过。
- 同一 production wrapper 的 deterministic synthetic acceptance 通过：12 次 wrapper forward、8 次 Smooth batch evaluation、sufficient/report/lineage reload 通过；synthetic classification 不具科学意义。focused tests=`7 passed`，py_compile/Ruff/diff-check 通过。
- dry-run 计数固定为 real TRAIN scientific reads=`0`、real DEV scientific reads=`0`、scientific forward=`0`、Smooth scientific evaluation=`0`；没有执行 V5 attempt-03、training、optimizer/backward/update、Stage B、generation、protected/test/sealed read 或 formal promotion。

该工程完成本身不构成 V5 scientific classification；attempt-03 结果已在上方独立记录。V5 mechanism 仍 unresolved，V6/Stage B、generation readiness 与 formal promotion 保持关闭，attempt-01/02 immutable evidence 保持不变。

## V5 latent-to-optical transfer-coupling diagnosis attempt-02（2026-09-11，INVALID；不重试）

按新的明确授权 `DIAGNOSE_V5_LATENT_ALIGNMENT_TO_OPTICAL_TRANSFER_COUPLING_V1 attempt-02`，在
`research/lightgen-ols-native-semantic-generation-v1`、HEAD=`9a72e0bcf2de12ab9a0caa7151f273de76768dc7` 上执行一次。attempt-01 未被修改。修复后的 preflight authority 字段绑定已通过静态检查，但真实入口随后因再次发现的 harness 调用错误停止：结果=`INVALID/invalid_harness`，classification=`INVALID_V5_TRANSFER_COUPLING_UNEXPECTED_EXECUTION_ERROR`。

- [`attempt-02 diagnosis report`](../outputs/20260911_v5/runs/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_v1_attempt_02/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_report.v1.json)，SHA-256=`0389608886eb7e8ac7c83e7560ba55447484548403d120c5209d4f73f55204d8`；
- [`attempt-02 diagnosis lineage`](../outputs/20260911_v5/runs/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_v1_attempt_02/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_lineage.v1.json)，SHA-256=`85fdcdb668974e545d65b740d0cb062048e109956397da4888071265a323e37f`；
- attempt-02 执行时入口 SHA-256=`233e98e7036f71c36ed4c83c97dc2c4d5478a5dbea93304cce00fca529d361cd`。

失败原因是入口调用 accepted V5 preflight 中不存在的 `_load_evaluators`；权威 preflight 实际提供 `_load_smooth`。报告确认 `diagnosis_executed=false`、`scientific_forward_count=0`、`diagnosis_started=false`；无 optimizer/backward/parameter update/training，无 TRAIN/DEV scientific read，无 protected/official-test/sealed read，无 semantic classification。Outer Gate=`V5_LATENT_TO_OPTICAL_TRANSFER_MECHANISM_RESOLVED` remains unresolved。

失败后仅将入口改为现有 `_load_smooth`、保持 evaluator 冻结并继续区分 checkpoint file SHA 与 state-dict/tree SHA；未改写 attempt-01/02 报告或 lineage，未重跑。当前入口修复后 SHA-256=`8ef136a8e369e0ab81b0fc16fe53cd660e2d92950d6def1cd9f9b373cd8d1571`。若继续，必须重新明确授权一个新 attempt；本 attempt 不支持任何 V5 transfer mechanism 分类，也不开放 V6、Stage B、generation 或 formal promotion。

## V5 latent-to-optical transfer-coupling diagnosis attempt-01（2026-09-11，INVALID；不重试）

按明确授权 `DIAGNOSE_V5_LATENT_ALIGNMENT_TO_OPTICAL_TRANSFER_COUPLING_V1`，在
`research/lightgen-ols-native-semantic-generation-v1`、HEAD=`9a72e0bcf2de12ab9a0caa7151f273de76768dc7` 上执行一次且仅一次入口诊断。synthetic exact-core preflight 已通过，但真实 scientific 入口在 authority resolution 阶段因新 harness 错误停止：结果=`INVALID/invalid_harness`，classification=`INVALID_V5_TRANSFER_COUPLING_HARNESS`。

- [`diagnosis report`](../outputs/20260911_v5/runs/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_v1_attempt_01/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_report.v1.json)，SHA-256=`49236400f58c8a15da6d87632a7601c7e936fd2fe1280be1be37713f03611eb0`；
- [`diagnosis lineage`](../outputs/20260911_v5/runs/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_v1_attempt_01/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_lineage.v1.json)，SHA-256=`74552a2a1fdb8d94cf30616dd51ebbee759d3c85bae7205cb0089046da43937d`；
- [`diagnosis entrypoint`](../scripts/research/diagnose_lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_v1.py) 在该 attempt 中的 SHA-256=`14a04ca9988e554629447404aa749ee0c68581c021055678c12686c41f0dc1cb`。

失败原因是 V5 scientific input 的 preflight authority 摘要实际位于 `accepted_preflight.input.sha256`，入口错误读取为 `accepted_preflight.sha256`，故在任何 scientific forward 前触发 `V5 input preflight binding changed`。报告确认 `diagnosis_executed=false`、`scientific_forward_count=0`、`diagnosis_started=false`；未执行 optimizer/backward/parameter update/training，未读取 TRAIN/DEV scientific data，未访问 protected/official-test/sealed，未产生 semantic classification。Outer Gate=`V5_LATENT_TO_OPTICAL_TRANSFER_MECHANISM_RESOLVED` remains unresolved。

失败后仅修复了入口的已定位字段绑定，未改写本次 immutable report/lineage，且没有重跑；未来若继续，必须由新的明确授权重新消费 one-shot diagnosis。该 attempt 不支持 `V5_SUBTHRESHOLD_SEMANTIC_MOVEMENT_SUPPORTED`、`V5_LATENT_DIRECTION_MISALIGNMENT_SUPPORTED`、`V5_LATENT_DISPLACEMENT_MAGNITUDE_LIMITED_SUPPORTED` 或 `V5_LATENT_TO_OPTICAL_TRANSFER_MIXED_OR_UNRESOLVED` 任一科学分类，也不开放 V6、Stage B、generation 或 formal promotion。

## V5 direct complex-OLS semantic-alignment scientific comparison（2026-09-11；历史 support snapshot）

按明确授权 `EXECUTE_LIGHTGEN_2D_COMPLEX_OLS_SEMANTIC_ALIGNMENT_V5`，在 branch=`research/lightgen-ols-native-semantic-generation-v1`、HEAD=`9a72e0bcf2de12ab9a0caa7151f273de76768dc7` 上完成且仅完成一次 matched CUDA comparison：control 与 candidate 各 300 个真实更新，checkpoint 为 100/200/300；无 retry、调参、预算扩展或下游阶段。

结果为 `VALID_COMPLETED/valid_completed`，classification=`V5_OLS_SEMANTIC_ALIGNMENT_NOT_SUPPORTED`；Outer Gate=`V5_OLS_SEMANTIC_ALIGNMENT_SUPPORTED` 未通过。

- [`scientific report`](../outputs/20260911_v5/runs/lightgen_2d_complex_ols_semantic_alignment_v5_scientific_attempt_01/lightgen_2d_complex_ols_semantic_alignment_v5_scientific_report.v1.json)，SHA-256=`41aeb6e5ae5d1736521b164f2c8e9267bc3687801ab99b7142692112f8d85ce8`；
- [`scientific lineage`](../outputs/20260911_v5/runs/lightgen_2d_complex_ols_semantic_alignment_v5_scientific_attempt_01/lightgen_2d_complex_ols_semantic_alignment_v5_scientific_lineage.v1.json)，SHA-256=`89e00a5f67dc082b0d2f37042a66a796cae71f14b1f676e8a27e94caaeed4c50`；
- [`scientific entrypoint`](../scripts/research/execute_lightgen_2d_complex_ols_semantic_alignment_v5_scientific.py)，执行时 SHA-256=`b48d9f8ae9f128d9b2f7563e5892be8b82d7dc973ef9f56196a0ccf4571fae0f`；
- [`scientific input`](../outputs/20260911_v5/runs/lightgen_2d_complex_ols_semantic_alignment_v5_scientific_attempt_01/lightgen_2d_complex_ols_semantic_alignment_v5_scientific_input.v1.json)，SHA-256=`91607325a08883479d49e69adac2d6a2c8ea6251ed86da5b45664aa0a7179b77`；
- [`paired bootstrap`](../outputs/20260911_v5/runs/lightgen_2d_complex_ols_semantic_alignment_v5_scientific_attempt_01/lightgen_2d_complex_ols_semantic_alignment_v5_paired_bootstrap.v1.pt)，SHA-256=`2ded3c474df42c9188304669150977fcd2fa30b26ba493b40be3863ad6e46772`；[`sufficient observations`](../outputs/20260911_v5/runs/lightgen_2d_complex_ols_semantic_alignment_v5_scientific_attempt_01/lightgen_2d_complex_ols_semantic_alignment_v5_sufficient_observations.v1.pt)，SHA-256=`82104162a4058b789fc62a6028ba2d064bb6104c3d07732c05c2b42f6af138d9`；两者均回读通过。

固定条件为 control=`L_image`、candidate=`L_image + 0.09573533218209243*L_proto`；只训练 `encoder.phase`，generator/100-port complex OLS/`T=I_100`/shared `W_OLS` 均冻结；TRAIN prototypes=`55,000`，`s_mu=0.00555040966719389`，DEV evaluation=`5,000`（每类 500），calibration/protected/test/sealed=`0`。运行环境为 RTX 3060 Laptop、compute capability=`8.6`、PyTorch=`2.5.1`、CUDA=`12.1`、grid=`2048`、batch=`10`、Adam `lr=1e-3`。

| arm | step-100 Smooth / `L_image` / OLS margin | step-200 Smooth / `L_image` / OLS margin | step-300 Smooth / `L_image` / OLS margin |
| --- | --- | --- | --- |
| control | `0.1040` / `0.0664125154` / `0.0020675604` | `0.1046` / `0.0658696759` / `0.0020476440` | `0.1044` / `0.0654365251` / `0.0020292227` |
| candidate | `0.1038` / `0.0668724473` / `0.0021049360` | `0.1042` / `0.0667107936` / `0.0021195124` | `0.1042` / `0.0665877977` / `0.0021336675` |

step-300 primary comparison：Smooth `Delta=-0.0002000000`，paired bootstrap `B=999`、seed=`271828`、95% CI=`[-0.0010,0.0004]`，per-class 最大下降=`-0.004`；OLS margin `Delta=+0.0001044448`，95% CI=`[0.0001008631,0.0001078032]`；DEV image-loss `Delta=+0.0011512727`，在 `+0.005` 容差内；两端 raw-CMOS panel 均 finite/structurally valid/nonconstant 且回读通过。因此 OLS margin、image-loss、per-class floor、raw panel 条件通过，但 Smooth `Delta>=0.05` 与其 CI 下界大于 0 两项未通过，整体 objective-correction gate 为 `NOT_SUPPORTED`。

candidate step-300 Smooth macro=`0.1042 < 0.937`，Stage-A firewall=`STAGE_A_FIREWALL_FAILED`；`stage_b_authorized=false`。scientific execution 的 forward/backward/update 已完成，但没有 Stage B、formal scale、prior、sampled generation、protected/test/sealed read、generation acceptance 或 formal promotion；`generation_readiness=not_ready`。该结果是 bounded objective-comparison evidence，不是 semantic generation 成功或下游授权。

## V5 direct complex-OLS semantic-alignment definition 与工程预检（2026-09-11，科学执行前 authority）

V5 semantic authority=`DEFINE_LIGHTGEN_2D_COMPLEX_OLS_SEMANTIC_ALIGNMENT_CONTRACT_V5` 已定义并通过 authority resolution，结果为
`READY/valid_completed`，classification=`DEFINE_LIGHTGEN_2D_COMPLEX_OLS_SEMANTIC_ALIGNMENT_CONTRACT_V5_READY`，未来科学 Outer Gate=`V5_OLS_SEMANTIC_ALIGNMENT_SUPPORTED`。
合同为
[`V5 semantic-alignment contract`](../outputs/20260911_v5/runs/lightgen_2d_complex_ols_semantic_alignment_contract_v5_attempt_01/lightgen_2d_complex_ols_semantic_alignment_contract.v5.json)，
SHA-256=`e2d57fa96e0a020abf23cbd889188d74d4893a6e9c48ac7a1d24d7156cf21b61`。

合同固定：V3 CONTROL step-300 encoder、有效 V4 step-300 fixed-OLS generator（只训练 `encoder.phase`，generator 与 OLS 冻结）、
100-port complex OLS、mandatory modal projection/re-emission、`T=I_100`、shared `W_OLS`、TRAIN-only ten prototypes、现有 gauge/distance，
以及 control=`L_image`、candidate=`L_image + lambda_proto*L_proto`。TRAIN 原型覆盖 `55,000` 行，45 对距离的中位尺度
`s_mu=0.00555040966719389`（距离范围=`0.0014081387780606747–0.01568867452442646`）。

按新的 preflight entrypoint 完成最终 attempt-05：结果为
`valid_completed/AUTHORIZABLE`，classification=`LIGHTGEN_2D_COMPLEX_OLS_SEMANTIC_ALIGNMENT_V5_AUTHORIZABLE`。
这只证明未来一次性 scientific comparison 的工程入口可执行，不是 V5 科学支持、Stage-A eligibility、Stage-B authorization、generation readiness 或 formal promotion。

- [`preflight input`](../outputs/20260911_v5/runs/lightgen_2d_complex_ols_semantic_alignment_preflight_v5_attempt_05/lightgen_2d_complex_ols_semantic_alignment_preflight_input.v1.json)，SHA-256=`9c734a76b5ceb03fe6405faadfaf7d2fa27eefe21d5a077ab3f0e46d926569f8`；
- [`preflight report`](../outputs/20260911_v5/runs/lightgen_2d_complex_ols_semantic_alignment_preflight_v5_attempt_05/lightgen_2d_complex_ols_semantic_alignment_preflight_report.v1.json)，SHA-256=`7fab2789b36491597462ce092bc98dc5784cc269245a2cab3b07a4878543c551`；
- [`preflight lineage`](../outputs/20260911_v5/runs/lightgen_2d_complex_ols_semantic_alignment_preflight_v5_attempt_05/lightgen_2d_complex_ols_semantic_alignment_preflight_lineage.v1.json)，SHA-256=`e5d72fce35784ea3c25fc98a5c4a7ad20538594f491f9065123a94f5892387dc`。

运行条件为 RTX 3060 Laptop、compute capability=`8.6`、PyTorch=`2.5.1`、CUDA build=`12.1`、2048 grid、batch=`10`、Adam `lr=1e-3`。
首个 TRAIN batch 的一次性梯度校准为 `g_image=0.0005046187151008359`、`g_proto=0.005270976801389226`，故固定
`lambda_proto=0.09573533218209243`；未更新前的 runtime state identity 保持不变。control/candidate 各完成真实 CUDA 三步，
encoder phase 均非零更新，generator 与 OLS 每步 unchanged；candidate prototype gradient 有限且非零，Smooth evaluator 无梯度。
两臂 checkpoint/resume 的 next batch、optimizer、runtime、raw endpoint 与 trace 差异均为 `0`。step-2 checkpoint 为：
control SHA-256=`2b3bde0e16e1224a03f6d887b5e66deeb533919163c4691be06ab89cc93274a9`，candidate SHA-256=`323d48d06c4fe45904c62e8c083ba832c54ba096958f45ce151c34b920b0f84e`。

synthetic exact-core acceptance 已覆盖 prototypes/scale/logits/loss/lambda、control/candidate、encoder-only 与 frozen firewall、raw CMOS、
checkpoint/resume、Smooth/margin/bootstrap/statistics、sufficient-observation reload 与 schema；其结果明确为 `scientific_meaning=none`，
不构成科学分类。production-grid no-data smoke 通过；TRAIN 读取=`60,000`、用于 prototype=`55,000`、两臂更新各=`30` 行；DEV scientific values=`0`，
protected/test/sealed=`0`。Windows NVRTC 通过当前进程 `sys.prefix\\bin` PATH repair，未修改机器级环境。

V5 scientific 300-step comparison 的一次性结果已记录于本文的历史科学区；本段仅保留 definition/preflight authority。`stage_b_authorized=false`、
`generation_readiness=not_ready`、`formal_promotion=false` 保持；不得因 preflight 或 scientific result 自动启动任何后续阶段。

## 当前主线，一页读完

> 本节保留 2026-09-11 的路线摘要；其中 `4L Linear` 的 2026-09-16 C3000 block 是本摘要时点的历史最新科学结果。2026-09-18 之后的当前结果、主线与授权以本文顶部 current block 为准；本节中的 V5 只表示历史 support route。

**唯一活动研究主线：输入与 OLS 对齐 LightGen 的光学语义生成原型。**
目标和长期约束见 [项目记忆](PROJECT_MEMORY.md#唯一活动研究主线lightgen-输入与-ols-对齐的语义生成原型)。
`label + 8D` 是正式兼容基线，不是本轮研究的输入目标；N3 是已有组件研究，不等于论文复现完成。

完整路线身份、`DATA_DIST`/latent/semantic/power taxonomy、objective matrix 与 claim 边界在本项目的
稳定导航和当前状态文档中分层维护；当前树为：
`MAINLINE=M0`；`SUPPORT=S1 complex-OLS MMD、S2 V3–V11 semantic diagnostics、S3 N3 producer、
S4 architecture/nonlinear/physical ablations`；`BASELINE=B0 label+8D`；`HISTORICAL=H1
Scaled-Chen、H2 v1 LDM teacher`。本页继续作为日期化数值、阻塞和停止边界的权威，不因注册表创建新增科学结论。

| 问题 | 当前答案 |
| --- | --- |
| 已经有什么 | 真实复 OLS 直接解码证据；已冻结的 100 维 Gaussian 振幅 producer；LightGen 对齐 input/100-port complex-OLS runtime；V2 Stage-A attempt-02 的完整 300-step trace；V3/V4 scientific 与诊断证据；V5 fixed-condition complex-OLS semantic-alignment comparison 的完整 300-step paired artifacts |
| 历史最新科学执行结果（截至 2026-09-16） | 4L Linear C3000 continuation=`VALID_COMPLETED` / `4L_LINEAR_CONTINUATION_PLATEAU_SUPPORTED`：`R=0.5790`、`Delta_R=+0.0016`、CI=`[-0.0002,+0.0034]`；`continue_to_4000=false`；Stage B/generation/formal promotion 均未开放 |
| 历史 V5 支持结果 | V5=`VALID_COMPLETED` / `V5_OLS_SEMANTIC_ALIGNMENT_NOT_SUPPORTED`：step-300 Smooth Delta=`-0.0002000`、CI=`[-0.0010,0.0004]`；OLS margin Delta=`+0.0001044448`、CI=`[0.0001008631,0.0001078032]`；image-loss Delta=`+0.0011512727`；candidate Smooth macro=`0.1042`，未达到 `0.937` firewall |
| 历史 V5 合同状态 | V5 contract=`READY`、preflight attempt-05=`AUTHORIZABLE`、scientific attempt-01=`VALID_COMPLETED/NOT_SUPPORTED`；V4 conditionality attempt-03=`VALID_COMPLETED/MIXED_OR_UNRESOLVED`；`V4_GENERATOR_CONDITIONALITY_RESOLVED` 未通过；Stage B/generation/formal promotion 均未开放 |
| 当前 V4 conditionality diagnosis | attempt-03=`VALID_COMPLETED`，`diagnosis_executed=true`；within/cross pairing mismatch 均有正的 paired MSE delta，但冻结规则未唯一支持 collapse、class 或 instance conditioning，classification=`V4_GENERATOR_CONDITIONALITY_MIXED_OR_UNRESOLVED`；attempt-01/02 的 pre-scientific INVALID 历史保持不变；不重试，`V4_GENERATOR_CONDITIONALITY_RESOLVED` 仍未解析 |
| 仍缺什么 | Stage-A semantic eligibility、完整 Stage B、post-training evaluation、latent compatibility、direct prior generation、semantic distribution、diversity、novelty；formal promotion/readiness 仍关闭 |
| 当前不能声称什么 | V3 objective correction 已被支持、provider-specific attribution 等于语义生成成功、V4 fixed-OLS recoverability 已被支持、V2 Stage-A 已达标、光学语义生成成功、直接 prior generation 支持、严格复现论文、generation readiness 或 formal promotion |
| 历史执行合同的 next-action 约束 | V5 scientific authorization 已一次性消费且不重试；`V5_OLS_SEMANTIC_ALIGNMENT_SUPPORTED` 未通过。该结果不自动开放 Stage B、training、generation 或 formal promotion；任何新 objective/weight/architecture 实验都需新合同与独立明确授权 |
| 历史结果对 next action 的约束 | 在固定 real OLS、固定 V4 step-300 generator、固定 TRAIN/DEV 角色与 `T=I_100`/shared `W_OLS` 下，冻结 prototype objective 产生了极小正向 OLS-margin 变化，但没有独立 Smooth 语义改进；不支持本 objective correction、Stage-A eligibility 或 Stage-B 资格，也不改变 `0.937` firewall |

## 最新 V2 Stage-A 科学训练与 warmup diagnosis（2026-09-11）

V2 Stage-A attempt-02 已按 corrected V2 authority 完成 300 个真实更新，结果为
`VALID_COMPLETED/valid_failed`，classification=`LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_V2_SEMANTIC_WARMUP_NOT_ESTABLISHED`。
报告为
[`lightgen_2d_semantic_complex_ols_training_v2_report.v1.json`](../outputs/20260911_v2/runs/lightgen_2d_semantic_complex_ols_training_v2_attempt_02/lightgen_2d_semantic_complex_ols_training_v2_report.v1.json)，
SHA-256=`de55a0f307675ded10aca5726f9ac09474633b2c9a5f8bfb1890075de53a530d`；lineage SHA-256=`899aa78daff4bbe2f70cf72e01080e323e24d791f4937ca35fefc8c9b30394dc`。

固定 checkpoint 为 100/200/300。DEV macro recall 为 `0.170 → 0.160 → 0.136`，DEV image loss 为
`0.0711633662 → 0.0689440685 → 0.0673580589`；Stage B、prior generation、heldout evaluation 和 promotion 均未执行。
对应的 bounded warmup diagnosis attempt-02 结果为 `VALID_COMPLETED`，primary diagnosis=
`SEMANTIC_WARMUP_FAILURE_IMAGE_OBJECTIVE_MISMATCH_SUPPORTED`；报告为
[`lightgen_2d_semantic_ols_warmup_failure_diagnosis_report.v1.json`](../outputs/20260911_v2/runs/lightgen_2d_semantic_ols_warmup_failure_diagnosis_v1_attempt_02/lightgen_2d_semantic_ols_warmup_failure_diagnosis_report.v1.json)，
SHA-256=`f05eb4e6dfd7c8c3ce1865ff8097fdef2dfa649b3e3dab3352f5282fa544f3da`；lineage SHA-256=`ef72fcaf191fc147d34fc460fb0d0f147614b2d696d677feae7d5d501f40b763`。

该诊断显示 encoder/generator 梯度有限且非零，零/非有限比例为 0；因此当前结论是“纯 image objective 与 semantic identity 不匹配”的有界支持证据，
不是已证明的容量不足、OLS 失效或 Stage-B 资格。TRAIN=`55,000`、DEV=`1,000`，calibration/protected/test/sealed 均为 0；V2 attempt-02 之后的 V3
objective-correction 定义取代了此前“下一步重复执行 V2 训练”的旧行动描述，历史段落仍保留事件记录。

## V3 Stage-A objective-correction contract（2026-09-11）

已物化一份 definition-only、不可自授权训练的 V3 合同：
[`lightgen_2d_semantic_complex_ols_objective_correction_contract.v3.json`](../outputs/20260911_v2/runs/lightgen_2d_semantic_complex_ols_objective_correction_contract_v3_attempt_01/lightgen_2d_semantic_complex_ols_objective_correction_contract.v3.json)，
SHA-256=`e537fbf4b856519a71e48c16a48c149fadd327e2ad12db84f64c953413fb53f6`。
V3 semantic authority ID=`DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_OBJECTIVE_CORRECTION_CONTRACT_V3`，classification=
`DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_OBJECTIVE_CORRECTION_CONTRACT_V3_READY`，Outer Gate=
`V3_SEMANTIC_OBJECTIVE_CORRECTION_SUPPORTED`。

合同冻结的 control 是 `L_control=L_image`；candidate 是 `L_candidate=L_image + lambda_feature*L_feature`，其中 provider 为 accepted
`FrozenMNISTEvaluator v1` 的 128D classifier-before embedding，固定 preprocessing 为 clamp `[0,1]` 后 adaptive average pool `28×28`，
L2 normalization 后使用 per-sample squared Euclidean distance。`lambda_feature` 只在共享初始状态的 TRAIN step-1 batch 上按
`min(10.0, max(0.1, g_image/g_feature))` 计算一次；两臂使用同一零 phase 初始化、batch schedule、Adam、300-step budget 与 100/200/300 checkpoints。
独立 `SmoothMNISTEvaluator v2.1` 只作预注册的 metric corroboration，不作训练 provider 或唯一 acceptance evidence。

Objective-correction 支持要求 step-300 的 Smooth macro recall 差值 `Delta_smooth >= 0.05` 且 999 次 paired class-stratified bootstrap 的 95% CI 下界 `>0`，
无 Smooth per-class recall 下降超过 `0.10`，candidate image loss 不超过 control `+0.005`，并且两臂 raw-CMOS endpoint panel 有限、结构有效且非恒定。
完整有效执行但未满足这些条件时返回 `V3_SEMANTIC_OBJECTIVE_CORRECTION_NOT_SUPPORTED`。这与 Stage-A eligibility 严格分离：Frozen evaluator macro recall
仍须达到且不能补偿的 `>=0.937`，本合同不授权 Stage B。

V3 合同要求未来工程 preflight 先验证真实 CUDA 三步更新、trainable/frozen firewall、provider input gradient、匹配初始化/顺序、disabled-term control、
detached-provider negative control 以及中断恢复 identity；在此之前不进行 scientific training。治理标志保持
`training_authorized=false`、`stage_b_authorized=false`、`generation_readiness=not_ready`、`formal_promotion=false`。

## 最新 V3 bounded engineering preflight（2026-09-11）

按上述 hash-bound V3 contract 执行了一次仅工程用途的 bounded preflight；最终 attempt-09 结果为
`valid_completed/AUTHORIZABLE`，classification=`LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_OBJECTIVE_CORRECTION_V3_AUTHORIZABLE`。
该结果只证明冻结的 control/candidate Stage-A 未来入口在当前环境中可执行，不证明 `V3_SEMANTIC_OBJECTIVE_CORRECTION_SUPPORTED`，不构成训练授权、Stage-A 资格或 Stage-B 授权。

输入、报告和 lineage 为：
[`preflight input`](../outputs/20260911_v3/runs/lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_attempt_09/lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_input.v1.json)，SHA-256=`631eb85792062f00e2e59344d091a6aac203225e05c56e1a4c27150859772d36`；
[`preflight report`](../outputs/20260911_v3/runs/lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_attempt_09/lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_report.v1.json)，SHA-256=`265e9b80582161872a207041cd5edc2279b6f72c5729aa6f5406f9646e354468`；
[`preflight lineage`](../outputs/20260911_v3/runs/lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_attempt_09/lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_lineage.v1.json)，SHA-256=`f600bf924c3494f0b348fccde19aa6d49d5b33923cff622d5a5722394cb08e31`。
执行入口为
[`execute_lightgen_2d_semantic_complex_ols_objective_correction_v3.py`](../scripts/research/execute_lightgen_2d_semantic_complex_ols_objective_correction_v3.py)。
step-2 control/candidate checkpoint 分别为
[`control`](../outputs/20260911_v3/runs/lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_attempt_09/lightgen_2d_semantic_complex_ols_objective_correction_v3_control_checkpoint.v1.pt)，SHA-256=`ac4d1e2521d4f265472aa4ba36673e19d42e16c8972b5bde24c24b5273a79c3a`，以及
[`candidate`](../outputs/20260911_v3/runs/lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_attempt_09/lightgen_2d_semantic_complex_ols_objective_correction_v3_candidate_checkpoint.v1.pt)，SHA-256=`b16011a5b443f16b31399c0e0b4d3a220402725b51e49eb1df52b449bc9b8c7f`。

当前设备为 RTX 3060 Laptop GPU、compute capability=`8.6`、PyTorch=`2.5.1`、CUDA build=`12.1`、VRAM=`6,441,926,656` bytes；
两臂均通过真实 CUDA `3` updates，grid=`2048`、batch=`10`、Adam lr=`1e-3`，且共享零 phase 初始状态、TRAIN schedule、optimizer manifest 与 RNG/sampler 起点。
lambda 只在首次 optimizer update 前计算一次：`g_image=0.0012350937723848974`、`g_feature=0.0631487965185555`、
`g_image/g_feature=0.01955846889373387`，按固定上下限得到 `lambda_feature=0.1`。control 的 `L_total=L_image`；candidate 的
`L_total=L_image+0.1*L_feature`。候选 semantic input gradient 三步均有限且非零（step 1/2/3=`0.0945075506/0.1344047379/0.1075679583`），
两组 trainable phase 均获得非零更新，modal projection、`T=I_100` 和 `W_OLS` 路径保持有效。

disabled-term control 通过；detached-provider negative control 保留有限 feature forward 但移除 semantic gradient，并被标记为应拒绝的
`invalid_harness` 路径；provider、independent evaluator 与 runtime non-phase state 均未更新。step-2 checkpoint 的 save/reload/resume 中，
next batch、sampler、RNG、lambda、provider identity 通过；control 恢复为 bitwise 相同，candidate 的 trace 最大差异为约 `5.62e-10`、
trainable parameter 最大差异为约 `1.65e-8`，在预检确定性容差内通过。该预检没有 scientific comparison、DEV semantic selection、300-step
scientific training、Stage B、scale/prior/generation、protected/test/sealed read 或 promotion；治理标志保持关闭。attempt-01/02 为 harness
适配修复，attempt-03 为 CUDA DLL 环境阻断，attempt-04 至 attempt-08 为恢复比较器修复；这些 immutable artifacts 均不覆盖 attempt-09 的最终结果。

## 最新 V3 Stage-A scientific objective comparison（2026-09-11）

按用户明确授权的唯一 scientific scope
`EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_OBJECTIVE_CORRECTION_V3`，从冻结的零 phase 初始状态执行一次
control/candidate 各 300 个真实 CUDA updates；没有重试、参数重选、Stage B、scale/prior/generation 或
protected/test/sealed read。结果为 `VALID_COMPLETED`，classification=`V3_SEMANTIC_OBJECTIVE_CORRECTION_NOT_SUPPORTED`。

主报告为
[`scientific report`](../outputs/20260911_v3/runs/lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_attempt_01/lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_report.v1.json)，
SHA-256=`8d7930449a1872e685ad39dafa499d89fdcc1d6bb5a9317e7ba06c554f54f31a`；lineage SHA-256=`00127edfad105e077342a8eb270447ea577cd4c48b816fb0435c0e6edbfc237d`。
执行入口为
[`scientific entrypoint`](../scripts/research/execute_lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific.py)，
SHA-256=`ca9eb8db3c7360f31da8d08ed6d0f61fa3bc61e8f0ce3de0f5a952803ff44ae4`。

step-300 primary endpoint 的有限结果为：

- Smooth macro recall：control=`0.103`，candidate=`0.107`，`Delta_smooth=+0.004`，低于预注册的 `+0.05`；
- paired class-stratified bootstrap（`B=999`，seed=`271828`）95% CI=`[0, 0.009]`，下界不大于 `0`；
- image loss：control=`0.0673580589`，candidate=`0.0726336428`，差值=`+0.0052755839`，超过 `+0.005` 容差；
- Smooth per-class 最大下降未超过 `0.10`，两臂 raw-CMOS endpoint panel 均 finite/structurally valid/nonconstant；
- Frozen evaluator macro recall：control=`0.136`，candidate=`0.283`，但 candidate 仍低于不可补偿的 `0.937` Stage-A firewall，故 firewall=`FAILED`。

因此，本次结果只支持“该冻结 single-term feature objective 在这组 300-step 条件下未达到预注册 objective-correction 支持标准”；
不支持“语义目标已解决”、Stage-A eligibility 或 Stage-B。6 个 checkpoint 与 6 个固定 `[1000,1,200,200]` raw-CMOS panel、bootstrap artifact
均已回读并通过 hash/shape/finite 核验；`training_authorized=false`、`stage_b_authorized=false`、`generation_readiness=not_ready`、
`formal_promotion=false` 保持。

## 最新 V3 objective-correction failure attribution（2026-09-11）

在 V3 scientific comparison 有效失败之后，按
`RESOLVE_V3_OBJECTIVE_CORRECTION_FAILURE_ATTRIBUTION_V1` 对既有 control/candidate step-300 checkpoint
执行了一次 forward-only、no-update 归因诊断；没有 optimizer、backward、光学参数更新、V3 retry、Stage B、scale/prior/generation 或
protected/test/sealed read。结果为 `VALID_COMPLETED`，Outer Gate=`V3_OBJECTIVE_FAILURE_ATTRIBUTION_RESOLVED`，唯一 classification=
`PROVIDER_SPECIFIC_ALIGNMENT_SUPPORTED`。该分类是对“冻结 provider 的提升未转移到独立 evaluator，且固定 latent diagnostic 未建立匹配 semantic gain”的有界归因，
不是语义生成成功、Stage-A eligibility 或 Stage-B 授权。

权威报告为
[`failure attribution report`](../outputs/20260911_v3/runs/lightgen_2d_semantic_complex_ols_objective_correction_failure_attribution_v1_attempt_05/lightgen_2d_semantic_complex_ols_objective_correction_failure_attribution_report.v1.json)，
SHA-256=`b3175da21e50448c5e6eeb0c2c36042f6b620c2510865c5059de26eb43277393`；
[`input`](../outputs/20260911_v3/runs/lightgen_2d_semantic_complex_ols_objective_correction_failure_attribution_v1_attempt_05/lightgen_2d_semantic_complex_ols_objective_correction_failure_attribution_input.v1.json)，
SHA-256=`0c4945ca91641e89de44af3c131106d5b8fa55d42dbe842244b57e1b40ced794`；
[`lineage`](../outputs/20260911_v3/runs/lightgen_2d_semantic_complex_ols_objective_correction_failure_attribution_v1_attempt_05/lightgen_2d_semantic_complex_ols_objective_correction_failure_attribution_lineage.v1.json)，
SHA-256=`21c095047ce119f69795cb543acb5ac5fac5199380c383024ec8e6638cda9be5`。

诊断固定读取 V3 control checkpoint SHA-256=`a42a43fdf99cf12d9abd1917a6d025fc2fcb9ecee39f0c05429dd8d7a8870079` 与 candidate checkpoint
SHA-256=`56ad62e41e0776f89d6c6d50fa5475b330659607e6630d28dc737ac6337e39d2`；两臂共享 `T=I_100`、`W_OLS`、modal projection 与 evaluator preprocessing。
固定 DEV 的 matched 结果为 Frozen macro=`0.136→0.283`、Delta=`+0.147`、paired bootstrap 95% CI=`[0.119, 0.17005]`；
独立 Smooth macro=`0.103→0.107`、Delta=`+0.004`、CI=`[0, 0.009]`；Frozen 128D consistency distance Delta=`-0.1388709251`、CI=`[-0.1496474, -0.1278142]`。

完整 2×2 cross-swap 的 Smooth macro 为：`E_control+G_control=0.103`、`E_candidate+G_candidate=0.107`、
`E_candidate+G_control=0.102`、`E_control+G_candidate=0.107`；四条 raw-CMOS panel 均 finite、结构有效且非恒定。
full-complex gauge `S(z)∈R^200` 的 participation effective rank 为 control=`15.5868`、candidate=`15.8340`，
between/within ratio 为 `0.959968→0.961896`；固定 TRAIN 每类前 100 个 source-index-sorted row 建 centroid 后，DEV nearest-centroid macro=`0.761→0.763`，
Delta=`+0.002`、paired bootstrap CI=`[-0.005, 0.010]`，因此 latent diagnostic 不支持 material semantic gain。

该 attribution 仅解析失败位置，不改变 Stage-A firewall：candidate Frozen macro=`0.283 < 0.937`，`stage_b_authorized=false`；后续不得自动重训、重试或晋级。

## 最新 V4 fixed-OLS generator-recoverability contract 与工程预检（2026-09-11）

基于 V3 scientific failure 与 `PROVIDER_SPECIFIC_ALIGNMENT_SUPPORTED` attribution，已定义一个且仅一个
固定 OLS 生成器恢复性实验：冻结 V3 CONTROL step-300 encoder 及其真实复数 `z∈C^100`，恢复原始 zero-phase
generator，只训练 `generator.phase`，优化目标固定为纯 `L_image`。100-port complex OLS、mandatory modal
projection/re-emission、`T=I_100`、shared `W_OLS`、2048 propagation grid、TRAIN order、Adam `lr=1e-3`
和 Stage-A `0.937` firewall 均保持不变；semantic provider 不进入优化路径，只作无梯度、无状态更新的评估隔离检查。

V4 semantic authority ID=`DEFINE_LIGHTGEN_2D_FIXED_OLS_GENERATOR_RECOVERABILITY_CONTRACT_V4`，结果为
`READY/valid_completed`，classification=`DEFINE_LIGHTGEN_2D_FIXED_OLS_GENERATOR_RECOVERABILITY_CONTRACT_V4_READY`，
Outer Gate=`FIXED_OLS_GENERATOR_RECOVERABILITY_SUPPORTED`。契约为
[`lightgen_2d_fixed_ols_generator_recoverability_contract.v4.json`](../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_generator_recoverability_contract_v4_attempt_01/lightgen_2d_fixed_ols_generator_recoverability_contract.v4.json)，
SHA-256=`d46740ded4db58d7d9f0caa8acaf40942d03dc3407d9611d24efcaa12d3036b8`。

契约绑定 V3 CONTROL step-300 checkpoint SHA-256=`a42a43fdf99cf12d9abd1917a6d025fc2fcb9ecee39f0c05429dd8d7a8870079`，
encoder.phase tensor SHA-256=`61648ce609ce4e62f890112faf3f3f91ffa5c018e630ec2d451a3d19c11ed9ca`；原始 zero-phase generator
由 fresh corrected-V2 runtime 重现，runtime state SHA-256=`0f90086cf26a179905a27a5a519f7599de256e4f4225f7798f24aeed086eb813`，
`generator.phase` tensor SHA-256=`68c468f955fb6606cd38278ff4ee4fb9f4a0c6c14337f59c72fc9be2eabb32d9`。V4 的 trainable surface
是仅 `generator.phase` 的 `20,000` 个参数；encoder `20,000` 参数、OLS frozen state `2,010,000` scalars 均冻结。

最终工程预检为 attempt-02，结果为
`valid_completed/AUTHORIZABLE`，classification=`FIXED_OLS_GENERATOR_RECOVERABILITY_PREFLIGHT_AUTHORIZABLE`：

- [`preflight input`](../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_generator_recoverability_preflight_v4_attempt_02/lightgen_2d_fixed_ols_generator_recoverability_preflight_input.v1.json)，SHA-256=`ef3388a575c1eb648baef1c60e8932d6d0d51805af09a784d68dc14e7739e626`；
- [`preflight report`](../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_generator_recoverability_preflight_v4_attempt_02/lightgen_2d_fixed_ols_generator_recoverability_preflight_report.v1.json)，SHA-256=`c2b0d852e1178915523ceee21172b10b8d462b6d0096ecf87c8443c410c7b7fb`；
- [`preflight lineage`](../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_generator_recoverability_preflight_v4_attempt_02/lightgen_2d_fixed_ols_generator_recoverability_preflight_lineage.v1.json)，SHA-256=`139fd231c73cb5d8f90e74508e6d8d13645b7a6c9c30d839ad7311cbba642da8`；
- [`step-2 checkpoint`](../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_generator_recoverability_preflight_v4_attempt_02/lightgen_2d_fixed_ols_generator_recoverability_preflight_step_2_checkpoint.v1.pt)，SHA-256=`00bd0af0f2c8bbcded39215bb6cf53bfc0fcd64efb952d4b693557ca0308dcec`；
- [`V4 preflight entrypoint`](../scripts/research/execute_lightgen_2d_fixed_ols_generator_recoverability_v4.py)，SHA-256=`8f99e098e121ab3dc8b667ddcd9c1dfbf57d1687fbfd8b87e166cf3bbb2d5b73`。

工程预检完成 3-step continuous generator-only trace（production grid=`2048`，batch=`10`，TRAIN-only）；generator
gradient norm=`0.0009246/0.0006487/0.0008156`，encoder gradient=`None`，generator update norm=`0.13147/0.10217/0.08960`，
encoder 与 OLS 每步均 unchanged，modal/W_OLS/T=I_100 provenance 均通过。canonical frozen-encoder z 与 cached-z replay
的最大差异为 `0`，raw replay 最大差异为 `0`；step-2 save/reload/resume 的 step-3 loss、gradient、update、raw、runtime
state 与 optimizer-state 最大差异均为 `0`。Frozen/Smooth provider 只做 TRAIN smoke 的 no-grad forward，feature shape 均为
`[10,128]`，状态未变且参数无梯度；protected/test/sealed 均未读取，DEV values 未消费。

attempt-01 仅作为 harness 修复历史，因 complex state comparator 的校验实现发出不完整复数比较警告，未作为最终证据；修复后
attempt-02 才是本段的 accepted preflight authority。V4 preflight 仍不是 scientific result：没有执行 300-step scientific run、
Stage-A semantic selection、Stage B、scale extraction、prior/generation、protected/test/sealed read 或 formal promotion；`0.937`
Stage-A firewall 不变，`training_authorized=false`、`stage_b_authorized=false`、`generation_readiness=not_ready`、
`formal_promotion=false` 保持。

## 最新 V4 fixed-OLS generator-recoverability scientific attempt-02（2026-09-11）

按独立明确授权 `EXECUTE_LIGHTGEN_2D_FIXED_OLS_GENERATOR_RECOVERABILITY_V4`，仅消费新的 scientific attempt-02；attempt-01
永久保留为 `BLOCKED/INVALID_HARNESS`（0/300），未被覆盖。attempt-02 在 bounded RNG/authority check 通过后完成精确
300 个真实 CUDA generator-only updates，结果为 `VALID_COMPLETED`，classification=`FIXED_OLS_GENERATOR_RECOVERABILITY_NOT_SUPPORTED`。

权威产物为：

- [`scientific input`](../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_generator_recoverability_scientific_attempt_02/lightgen_2d_fixed_ols_generator_recoverability_scientific_input.v1.json)，SHA-256=`8acd28ef99983e97d37f1d97a7ca39bf50d159ad358da520dbdfdae04a4ba2f9`；
- [`scientific report`](../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_generator_recoverability_scientific_attempt_02/lightgen_2d_fixed_ols_generator_recoverability_scientific_report.v1.json)，SHA-256=`fdfeb8d800a2d806f3ecc56503ef8d66cc68f06de6cd9e47aa4eab7c93ca2784`；
- [`scientific lineage`](../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_generator_recoverability_scientific_attempt_02/lightgen_2d_fixed_ols_generator_recoverability_scientific_lineage.v1.json)，SHA-256=`6089bc41797580aa95149b0478a0e25190a450e00ec4a47c886b756f0187f9e6`；
- 修复后 scientific entrypoint SHA-256=`a2a54556348deddc21a5ae0caf5c1343c9f6736f73d211aa33a1958ffb33375b`；
- attempt-01 invalid report SHA-256=`9887f93bd942c9e4cf21017aa5e443864bc4bcc8862390bf41023f13184f62b2`，lineage SHA-256=`70780327719c6314c7515db3386d23d806294b5be5021baca6f81225b875ec81`。

RNG reopening proof 通过：captured/expected scientific initial digest 均为
`6241d4343843975f8306769ff6ff05c0cb6af5b86956b545892592cf0e65ea30`，`cuda_available=true`，CPU 与 CUDA state identity
均通过；fresh runtime state SHA-256=`0f90086cf26a179905a27a5a519f7599de256e4f4225f7798f24aeed086eb813`，zero-phase generator tensor
SHA-256=`68c468f955fb6606cd38278ff4ee4fb9f4a0c6c14337f59c72fc9be2eabb32d9`，V3 CONTROL encoder tensor SHA-256=
`61648ce609ce4e62f890112faf3f3f91ffa5c018e630ec2d451a3d19c11ed9ca`。

冻结运行条件为 RTX 3060 Laptop GPU、compute capability=`8.6`、PyTorch=`2.5.1`、CUDA build=`12.1`、grid=`2048`、
batch=`10`、Adam `lr=1e-3`。仅 `generator.phase` 的 `20,000` 参数可训练；encoder、OLS、runtime non-generator state 与两个
evaluator 均保持 unchanged，semantic provider gradient=`false`。

固定 DEV 选择上的 trajectory 为：

| step | TRAIN image loss | DEV image loss | Smooth macro |
| --- | ---: | ---: | ---: |
| 100 | 0.0656021386 | 0.0693966536 | 0.105 |
| 200 | 0.0654420778 | 0.0682117225 | 0.104 |
| 300 | 0.0698628575 | 0.0673444915 | 0.103 |

step-300 相对 V3 CONTROL：Smooth=`0.103→0.103`，delta=`0`；paired bootstrap `B=999`、seed=`271828`、95% CI=`[0,0]`；
per-class delta 全部为 `0`，最大 degradation=`0`；V3 control DEV image loss=`0.0673580589`，V4=`0.0673444915`，
delta=`-0.0000135674`。raw-CMOS `finite/shape_valid/structured/nonconstant` 全部为 `true`。

冻结 Gate 判定：Smooth improvement `0 >= 0.05`=`FAIL`；bootstrap lower `0 > 0`=`FAIL`；per-class preservation=`PASS`；
image-loss tolerance `-0.0000135674 <= 0.005`=`PASS`；raw-CMOS validity=`PASS`。因此唯一 Outer Gate 返回
`FIXED_OLS_GENERATOR_RECOVERABILITY_NOT_SUPPORTED`，不得 retry。

独立 Stage-A firewall 为 Smooth DEV macro=`0.103 < 0.937`，结果=`STAGE_A_FIREWALL_FAILED`。TRAIN optimization rows=`55,000`、
DEV evaluation selection=`1,000`（DEV role=`5,000`），calibration/protected/test/sealed=`0`；没有 Stage B、formal scale
extraction、prior shaping、sampled-OLS generation、generation acceptance 或 formal promotion。治理保持
`training_authorized=false`、`stage_b_authorized=false`、`generation_readiness=not_ready`、`formal_promotion=false`。

## 研究 provenance 治理：HASH_ON_TRUST_BOUNDARY_V2（2026-09-11）

本轮仅更新研究工作流治理与 V4 conditionality diagnosis 入口校验，没有执行新的 scientific forward、training 或生成。
现行规则是：普通研究交接以 semantic authority ID、producer task/attempt 与 Git branch/HEAD 为默认身份；完整 child
SHA-256 继续保留在 machine manifest/registry、formal evidence、checkpoint transfer 与 write-once artifact boundary，
但只在真实 consumer 要求且身份域相同的情况下核验。

`file_byte_sha256`、`state_dict_tree_sha256`、`tensor_content_sha256`、`semantic_state_identity` 与 `manifest_sha256` 不可
互换。未被 consumer 要求的关系标为 `NOT_REQUIRED` / `NOT_APPLICABLE`；跨域摘要不同不构成 `INVALID_HARNESS` 或
`HARD_BLOCKER`。只有同域、明确要求的 authoritative Hash mismatch 才保持 fail-closed rejection。

V4 conditionality diagnosis attempt-01 的 immutable 结果仍为 `INVALID`，`diagnosis_executed=false`、scientific forward=`0`、
TRAIN/DEV diagnostic reads=`0`，没有科学结论。其入口已移除 Smooth checkpoint file SHA 与 loaded state_dict/tree SHA 的跨域
equality gate，改为检查既有 Smooth v2.1 authority 的 role、expected checkpoint file identity，以及现有 loader 的 loadability/schema
contract；两者的 state-tree digest 仅分域记录。focused regression tests 已验证该行为，未重试诊断，
`V4_GENERATOR_CONDITIONALITY_RESOLVED` 仍未解析。

随后在新治理 HEAD `1338ef5bad1936edd1f98fb3577933ed9bbd524a` 上按独立授权执行 V4 conditionality diagnosis attempt-02 一次。
该次在 authority resolution 阶段被不可变 V4 contract 仍要求旧 HEAD `383197a930dd48f98294120f8cf867444f994f85` 拒绝，
因此保持 `INVALID`：未写入 diagnosis input、scientific forward=`0`、TRAIN/DEV diagnostic reads=`0`，没有 semantic
classification。report SHA-256=`db2d6fcbe98cf7a0b732d6c9ae09d69931e63e77c7250149cdd3ce44d479d29c`；lineage
SHA-256=`8f23bca691b459d1ee24e78dd3e6131e3bb3cde1139a536ab52fcb0373ce09ba`。
按 one-shot 授权不 retry；`V4_GENERATOR_CONDITIONALITY_RESOLVED` 保持 unresolved，training、Stage B、generation 与 formal
promotion 均未开放。

本轮后续治理修复已在当前 HEAD `1338ef5bad1936edd1f98fb3577933ed9bbd524a` 完成并提交：历史 V4 contract 的
`git.head` 现在只作为 producer/contract provenance 记录，V4 diagnosis consumer 单独记录自己的 branch/HEAD；
缺少显式 `consumer_head_exact_required=true`（或等价 exact-source-lock）时，不再把 producer/consumer HEAD 差异判为
invalid。V4 contract、attempt-01 与 attempt-02 历史 artifacts 均未修改；本轮只运行 focused governance tests、compile、
Ruff 与 diff check，没有执行 V4 diagnosis、scientific forward、training、Stage B 或 generation。

## 最新 V4 generator conditionality diagnosis attempt-03（2026-09-11）

按独立授权 `EXECUTE_V4_GENERATOR_CONDITIONALITY_DIAGNOSIS_ATTEMPT_03`，在 consumer HEAD
`054026ad3b44558b108888178ac2fb58aebb2f52` 上对历史 V4 contract producer HEAD
`383197a930dd48f98294120f8cf867444f994f85` 执行一次且仅一次 forward-only diagnosis。producer/consumer HEAD
分离记录，`consumer_head_exact_required=false`；branch、HEAD 与工作区 status 在执行前后保持不变。

结果为 `VALID_COMPLETED`，但 primary classification=`V4_GENERATOR_CONDITIONALITY_MIXED_OR_UNRESOLVED`。
冻结的 matched endpoint reconstruction MSE mean/median=`0.0673444926/0.0652310392`；raw CMOS finite、
shape-valid、structured、nonconstant；Smooth target-label macro recall=`0.103`，仅作 descriptive evidence。

- within-class permutation：`Delta_within` mean/median=`0.0056112921/0.0023668033`，paired bootstrap 95% CI=
  `[0.0048836686, 0.0063864972]`，`P(L_matched < L_within)=0.632`，raw RMS change mean=`0.0961877042`；
- cross-class permutation：`Delta_cross` mean/median=`0.0093417037/0.0059634257`，paired bootstrap 95% CI=
  `[0.0083926659, 0.0102811366]`，`P(L_matched < L_cross)=0.720`，raw RMS change mean=`0.1161952598`；
- class contrast `Delta_class=Delta_cross-Delta_within` mean=`0.0037304116`，95% CI=
  `[0.0030554547, 0.0044253468]`。

因此错误 OLS pairing 会改变重建误差和 raw output，但预注册 decision checks 均为 false：collapse=false、
class_conditioning=false、instance_conditioning=false；不能强行归因为 class-level 或 instance-level
conditioning。Smooth target-label macro 在 matched/within/cross 为 `0.103/0.103/0.098`，independent semantic
rendering 仍未建立。TRAIN-only template attribution 的 nearest-class counts 为
`[1,501,28,0,123,297,7,25,7,11]`，class-8 fraction=`0.007`，global-mean distance mean=`0.0171787328`；
该结果不属于 generation-readiness metric。

执行边界：CUDA:0（NVIDIA GeForce RTX 3060 Laptop GPU，compute capability `8.6`，torch `2.5.1` / CUDA `12.1`），
production grid=`2048`、batch=`10`、modal projection=true、`T=I_100`、shared `W_OLS`、canonical replay max
absolute difference=`0`。TRAIN template reads=`55,000`，DEV role=`5,000`、selection=`1,000`，calibration/
protected/test/sealed=`0`；optimizer、backward、optical update、retraining、Stage B、scale、prior、generation
与 formal promotion 均为 NO。

Artifacts：report SHA-256=`93623cdc1d91f7418afd06ef49e53677c8f0c001801d8ae99c97aae738902e50`，lineage
SHA-256=`2ac4a8d23d5bc0228032597f94d541e12f7ebd8a60c501cb6b1313ec06c47aea`，input SHA-256=
`604cb8cef33e64e51a1e95a70fc5f3e3fe751849cbbdef0cddd1fd680f597b7b`。`V4_GENERATOR_CONDITIONALITY_RESOLVED`
仍未解析；attempt-01/02 immutable INVALID history 不变，不执行 retry。

## 最新 OLS class-direction transfer diagnosis attempt-01（2026-09-11）

按独立授权 `DIAGNOSE_COMPLEX_OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_V1`，仅建立并启动一次
forward-only/no-update diagnosis。当前 consumer HEAD=`054026ad3b44558b108888178ac2fb58aebb2f52`，branch
匹配授权；输出目录为 write-once attempt-01。入口在 authority/runtime loading 阶段调用了当前 helper 中不存在的
`_load_evaluators`，因此在任何 TRAIN/DEV 科学值读取、encoder forward、generator forward 或 Smooth evaluation
之前停止。结果为 `INVALID/invalid_harness`，classification=`INVALID_OLS_CLASS_DIRECTION_TRANSFER_UNEXPECTED_EXECUTION_ERROR`；
该分类是 harness 状态，不是四类 scientific class-direction classification。

本次 `diagnosis_executed=false`、scientific forward=`0`、TRAIN/DEV diagnostic reads=`0`、input 未写入；
optimizer、backward、optical update、training、V4 retry、Stage B、scale/prior/sampled generation、protected/test/sealed
read 与 formal promotion 均为 NO。由于 one-shot 规则，attempt-01 不修复后重跑；`OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_RESOLVED`
没有科学判定，V4 `V4_GENERATOR_CONDITIONALITY_RESOLVED` 仍 unresolved。

权威报告为
[`class-direction transfer report`](../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_v1_attempt_01/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_report.v1.json)，
SHA-256=`25f2fc26a0aee688721e6ae9a2229db52a0b5d4b101cc0f9bbbf254e7dc2bae7`；lineage 为
[`class-direction transfer lineage`](../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_v1_attempt_01/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_lineage.v1.json)，
SHA-256=`bffda2cc3210fa7fad63d5d9df9af1971ea2037a8288831be97333f810fec73c`。入口实现
[`diagnosis entrypoint`](../scripts/research/diagnose_lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_v1.py)
在执行时的 SHA-256=`95b85722fff33c09da83ac56024a7faa381c22f4cc24a83984d72fcf4a500697`。没有改写既有 V3/V4
contract、checkpoint、scientific report 或 INVALID history；任何未来修复/重开都需要新的明确授权与新的 attempt。

## 最新 OLS class-direction transfer diagnosis attempt-02（2026-09-11）

用户随后明确授权 `DIAGNOSE_COMPLEX_OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_V1` scientific attempt-02 一次且仅一次，
consumer HEAD=`a7cd8e082ad1b09c16c5c70fb38f0459cc583779`，入口已使用 canonical V4 evaluator loader。authority resolution 后，入口在
CUDA device preflight 的 complex/FFT probe 停止：底层 `nvrtc-builtins64_121.dll` 无法打开，故结果为
`INVALID/invalid_harness`，不是四类 class-direction scientific classification。

本次 `diagnosis_executed=false`、scientific forward=`0`、TRAIN/DEV scientific reads=`0`、diagnosis input 未写入；optimizer、
backward、optical update、training、V4 retry、Stage B、scale/prior/sampled generation、protected/test/sealed read 与 formal promotion
均为 NO。attempt-02 已按 one-shot 规则消费，不重试；`OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_RESOLVED` 仍 scientifically
unjudged，`V4_GENERATOR_CONDITIONALITY_RESOLVED` 继续 unresolved。

权威产物为
[`class-direction transfer report`](../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_v1_attempt_02/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_report.v1.json)，
SHA-256=`e35b3f00de4b3dc48394a384c91627169a4c835809e2951211eee0ebd67243c2`；以及
[`class-direction transfer lineage`](../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_v1_attempt_02/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_lineage.v1.json)，
SHA-256=`c70a8fef611e683da7e4b245f48b0daf95ac396f209608a2d041cca208c4cc64`。执行时入口工作树 SHA-256=`59e7a882e324378ab2011d940f979d7a7f24ff07139a19a9ab65b34da74f835d`；
attempt-01 immutable report/lineage 与其 harness 修复历史均未改写。

## class-direction 运行时工程 readiness 核查（2026-09-11）

未重试 attempt-02，也未执行 attempt-03。隔离核查确认 `_cuda_device_facts` 的失败操作是 CUDA complex `torch.exp(complex_probe)`；
complex allocation、`torch.polar` 与 complex FFT 本身均可通过。随后在不打开任何 TRAIN/DEV、calibration、protected、test 或 sealed 数据角色的情况下，
使用真实 LightGen-aligned `W_OLS → T=I_100 → 2-layer generator → output relay → raw CMOS`、production grid=`2048` 做一次 `[2,100]`
合成 complex OLS 前向：未修复环境首先在实际 generator complex path 报同一 `nvrtc-builtins64_121.dll`，因此原问题是 CUDA runtime DLL 搜索路径阻断，
不是非代表性的 device-facts probe。

该 DLL 已存在于当前 PyTorch 环境的 `sys.prefix/bin`；入口现仅在当前 Python 进程把该目录加入 `PATH`，不下载、不替换 DLL、不重装 PyTorch/CUDA，
缺失时仍 fail-closed 为 `CUDA_RUNTIME_ENVIRONMENT_BLOCKER_CONFIRMED`。修复后完整 readiness 顺序通过：authority resolution、CUDA facts、V4 step-300
checkpoint/runtime assembly、Frozen/Smooth evaluator load 与真实 2048 合成前向；raw CMOS=`cuda:0/[2,1,200,200]/float32`、finite/nonconstant，
modal projection/re-emission、`T=I_100`、两层 generator、state unchanged 与 gradients-none 均通过。该核查在科学数据边界前停止，
`scientific_forward_count=0`、TRAIN/DEV/protected/test/sealed reads=`0`，不产生任何 class-direction scientific classification。

新增回归覆盖：NVRTC 失败映射为 blocker、环境 DLL 缺失 fail-closed，以及 entry readiness 在 scientific data 前停止；focused tests=`8 passed`（含治理回归），
`py_compile`、Ruff 与 `git diff --check` 均通过。`OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_RESOLVED` 仍未判定，
`V4_GENERATOR_CONDITIONALITY_RESOLVED` 仍 unresolved；不构成 training、Stage B、generation readiness 或 formal promotion。

本轮从 HEAD=`a7cd8e082ad1b09c16c5c70fb38f0459cc583779` 完成 focused commit
`9f239d9dd1971872a7d5b2345ae11130574d48e4`（`research: add CUDA runtime readiness smoke`）；当前工作树中既有状态文档修改与 7 个此前未跟踪研究脚本保持未提交，未被本轮提交或清理。

## 最新 OLS class-direction transfer diagnosis attempt-03（2026-09-11）

用户明确授权一次且仅一次
`DIAGNOSE_COMPLEX_OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_V1_ATTEMPT_03`，入口先按已验证的
authority resolution、CUDA facts、V4 step-300 runtime/evaluator assembly 与 production-grid synthetic readiness path 执行。
自绑定从旧的 pre-commit HEAD=`a7cd8e082ad1b09c16c5c70fb38f0459cc583779` 最小重绑定到授权的当前 consumer HEAD=
`9f239d9dd1971872a7d5b2345ae11130574d48e4`；这是入口 provenance 修正，不改变 intervention、数据角色或统计规则。

readiness 通过后，科学阶段实际执行了 `5,800` 次 forward（TRAIN encoder=`5,500`，DEV canonical/replay/shifted 路径已进入），
但在计算 raw-response summary 时，入口调用 V4 helper 的不存在属性 `_raw_summary`，抛出
`AttributeError: module 'lightgen_v4_conditionality_helper_for_class_direction' has no attribute '_raw_summary'`。
因此结果为 `INVALID/invalid_harness`，classification=`INVALID_OLS_CLASS_DIRECTION_TRANSFER_UNEXPECTED_EXECUTION_ERROR`；
这是 partial scientific execution 的 harness 失败，不是四类 class-direction scientific classification，也不是科学 Gate failure。
raw、template、Smooth bootstrap、十个 class-pair summaries、broad/pair-dominated 与 Outer Gate 均未形成有效报告值。

不可变产物为：

- [`input`](../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_v1_attempt_03/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_input.v1.json)，SHA-256=`7b73020e355badbc279b2b1edfe346bdaef6019620bf5f2dc467b789dd631841`；
- [`report`](../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_v1_attempt_03/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_report.v1.json)，SHA-256=`a0b1032428c78c9f02b7db005732886186c1f68d3335c35602d1d01d2f6c025c`；
- [`lineage`](../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_v1_attempt_03/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_lineage.v1.json)，SHA-256=`03eb6a38d010a59de06ad18274ef186fc847c8fe94deb6553646389fc28f4876`。

Input manifest 记录的数据边界为 source=`60,000` rows loaded、TRAIN=`55,000`（centroid/template）、DEV role=`5,000`、固定 selection=`1,000`；
calibration/protected/official test/sealed 均为 `0`，未读取或打开 forbidden roles。device target 为 CUDA:0、RTX 3060 Laptop、compute capability=`8.6`、
PyTorch=`2.5.1`、CUDA build=`12.1`、grid=`2048`；`T=I_100`、shared `W_OLS`、modal projection/re-emission 与 frozen V4 generator 路径保持固定。
全程 `optimizer_created=false`、`backward_executed=false`、`optical_update_executed=false`、`training_executed=false`，并且
Stage B、scale/prior/sampled generation、formal promotion 均为 NO；`training_authorized=false`、`stage_b_authorized=false`、
`generation_readiness=not_ready`、`formal_promotion=false`。

attempt-03 已按 one-shot 规则消费，不重试、不在本轮修复后继续执行；`OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_RESOLVED`
保持 unresolved/unjudged，`V4_GENERATOR_CONDITIONALITY_RESOLVED` 仍 unresolved。任何修复或新诊断都必须先定义新的 bounded contract、
取得独立明确授权并新建 attempt；不得把本次 partial forward 当作方向性科学证据或开放 Stage B/generation。

## OLS class-direction transfer harness-failure replication（2026-09-11）

用户随后独立授权
`REPLICATE_COMPLEX_OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_AFTER_HARNESS_FAILURE_V1`，仅执行一次冻结语义的
forward-only/no-update replication。该 replication 不是 attempt-03 retry；attempt-03 的 `INVALID/invalid_harness`、`scientific_forward=5800`
与 `ATTEMPT03_NO_FORWARD_RECOVERY_UNAVAILABLE` 保持 immutable，未被读取改写或覆盖。replication 使用稳定化 runner consumer HEAD=
`9a72e0bcf2de12ab9a0caa7151f273de76768dc7`，starting/ending HEAD 相同，worktree 既有 dirty paths 未清理。

唯一结果为 `VALID_COMPLETED`，四类 scientific classification 为
`OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_SUPPORTED`，Outer Gate=`OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_RESOLVED`。
前置 authority/runtime/evaluator resolution、CUDA readiness、真实 `2048` production-grid synthetic smoke 与 fresh write-once output path 均通过。
设备为 CUDA:0、RTX 3060 Laptop、compute capability=`8.6`、PyTorch=`2.5.1`、CUDA build=`12.1`；runtime 使用 modal projection/re-emission、
`T=I_100`、shared `W_OLS`、固定 V4 step-300 generator，canonical replay max absolute difference=`0`。

数据边界严格为 source=`60,000` rows loaded、TRAIN=`55,000`（complex centroid/image template）、DEV role=`5,000`、固定 selection=`1,000`
（每类 `100`）；calibration/protected/official test/sealed 均为 `0`。scientific forward=`5,800`（TRAIN encoder=`5,500`、DEV
canonical=`100`、replay=`100`、shifted=`100`）；optimizer/backward/optical update/training/Stage B/scale/prior/sampled generation/protected/test/sealed/formal
promotion 均为 NO。

冻结 intervention 为 `t=(y+1) mod 10`、`z_shift=z-mu_y+mu_t`、`alpha=1.0`、无 renormalization。z、z_shift、centroid direction、raw base/shifted
均 finite；z norm mean=`0.0001486583`，z_shift norm mean=`0.0001123891`，`||mu_t-mu_y||` mean=`0.0000797999`。
raw absolute RMS change mean/median=`0.1127222995/0.1075329755`，relative mean/median=`0.8655350143/0.8446657972`；raw shape=`[1000,1,200,200]`，
finite、structured、nonconstant 均为 true。

Template 方向性：Delta mean=`0.0035271140`，median=`0.0021154392`，paired class-stratified bootstrap（`B=999`、seed=`271828`）95% CI=
`[0.0029781869,0.0040915711]`，fraction >0=`0.600`，primary pass=true。Smooth 方向性使用
`SmoothMNISTEvaluator v2.1 canonical pre-softmax logits`：Delta mean=`0.8398663402`，median=`1.4927580953`，CI=
`[0.6562327564,1.0271191061]`，fraction >0=`0.605`，primary pass=true。Smooth source prediction rate=`0.103→0.109`、target prediction rate=
`0.099→0.150`；target rank improvement（base minus shifted）mean=`0.398`，source rank change（shifted minus base）mean=`0.454`。

十个固定 pair 的 `n / mean Delta_template / positive-template fraction / mean Delta_smooth / positive-Smooth fraction` 为：

| pair | n | template mean | template + | Smooth mean | Smooth + |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0→1 | 100 | 0.0147247661 | 0.70 | 3.1643352509 | 0.75 |
| 1→2 | 100 | -0.0025137303 | 0.21 | -3.3669118881 | 0.03 |
| 2→3 | 100 | 0.0040614408 | 0.80 | 1.8833580010 | 0.80 |
| 3→4 | 100 | 0.0040716604 | 0.74 | 0.6570326090 | 0.54 |
| 4→5 | 100 | 0.0051244213 | 0.92 | 5.2552509308 | 0.95 |
| 5→6 | 100 | -0.0011038021 | 0.31 | -0.0760735869 | 0.48 |
| 6→7 | 100 | 0.0070621986 | 0.75 | 2.4041686058 | 0.75 |
| 7→8 | 100 | -0.0060858566 | 0.05 | -8.5505752563 | 0.02 |
| 8→9 | 100 | 0.0063659586 | 0.82 | 5.3506207466 | 0.97 |
| 9→0 | 100 | 0.0034887329 | 0.70 | 1.6340352297 | 0.76 |

正向 pair means 为 template=`7/10`、Smooth=`7/10`，满足 frozen broadness rule，解释为 `broad` 而非 pair-dominated；因此只产生上述一个
scientific classification，不改变任何 metric、threshold 或 classification rule。

本次 sufficient-observation artifact 已在 bootstrap/classification/report assembly 前 write-once 写入并回读验证：
[`sufficient observations`](../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_v1_attempt_01/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_sufficient_observations.v1.pt)，
schema=`lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_sufficient_observations.v1`，SHA-256=`0cce9236ef8ad79cb4898b4ed9c3dec121d50e9060d7dffdbf1bcb4274f0e87e`。
验证包括 `1000` samples、每类 `100`、selection identity hash、target rule、12 个 observation tensor 的 shape/set/hash 与 reload 一致；后续 recovery
不得重新执行 optical scientific forward。原始输出、bootstrap、report 和 lineage 为：
[`raw outputs`](../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_v1_attempt_01/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_raw_outputs.v1.pt)
SHA=`e985f6f030591a1aa38d33517d35da6c831806e121c9f73c3162395c23336d00`；
[`bootstrap`](../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_v1_attempt_01/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_bootstrap.v1.pt)
SHA=`1137d97ecf35fb383bfcf216a21cb4ef78a262c64c06c8795e15d5a93d6139b5`；
[`report`](../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_v1_attempt_01/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_report.v1.json)
SHA=`798e5af37f9d5e477fc395bd8230d17f25a20e56b0371f7376cfd1fbb7189883`；
[`lineage`](../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_v1_attempt_01/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_lineage.v1.json)
SHA=`d81c3829c8ae911d4aa0fd7620fca06d6dbecc16b3ee0292e024af5a76fa4496`。显式包含 replication identity、attempt-03 关系、历史事实与
FACT / MODEL_ASSUMPTION / DESIGN_DECISION taxonomy 的 write-once envelope 为：
[`replication report`](../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_v1_attempt_01/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_report.v1.json)
SHA=`f5b1b252b5815578fd09367258df66e7ba8f3ea008fc889cb30c23ec8d18c249`；
[`replication lineage`](../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_v1_attempt_01/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_lineage.v1.json)
SHA=`7a75ddb6008a65057c86e658104b6ccf61b14bd9091382f2101d55dae70440e7`。

FACT / MODEL_ASSUMPTION / DESIGN_DECISION 分离如下：

- FACT：本节列出的输入角色、forward count、finite/shape/structured 检查、两类 Delta、bootstrap CI、pair summaries 与治理标志是本次 replication 的测量；attempt-03 已进入科学 forward 后因缺失 `_raw_summary` 失败，且无 sufficient observations，不能被本次数据追溯重建。
- MODEL_ASSUMPTION：固定 OLS/generator 对 centroid replacement 的 broad template 与 Smooth 方向性响应可解释为 class-direction transfer evidence；它不证明 generator capacity、class-vs-instance conditionality 已解析、语义生成成功或 Stage-A/Stage-B eligibility。
- DESIGN_DECISION：`t=(y+1) mod 10`、`alpha=1`、`z-mu_y+mu_t`、full complex `C^100` centroid convention、无 renormalization、固定 template/Smooth/bootstrap/statistics/classification。

未解析机制仍包括 V4 generator conditionality（`V4_GENERATOR_CONDITIONALITY_RESOLVED` 仍 unresolved）、generator capacity/layer sufficiency 与 Stage-A semantic eligibility；
本 classification 不授权 training、Stage B、generation、scale extraction 或 formal promotion。

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
修复过程也保留为不可覆盖的历史 artifacts。本段末尾“下一动作是
`EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V2`”仅是该 preflight 时点的历史记录；V2 Stage-A attempt-02 已完成，当前动作以顶部
V3 objective-correction contract 为准，且不得自动开始。

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
