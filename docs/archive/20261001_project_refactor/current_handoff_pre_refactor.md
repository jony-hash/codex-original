# 当前研究交接（2026-10-01；执行规则简化审查）

当前执行政策以 [AGENTS.md](../../AGENTS.md#current-lightgen-research-policy-2026-10-01) 为准：取消墙钟上限和 hash 准入绑定，普通工程修复沿用任务授权，保留固定科学工作量、数据隔离及硬件安全。历史执行器的硬编码检查尚待迁移；本轮未启动新科学执行。审查在 CPU 复现了传播缓存的 inference-mode→backward 故障，未创建 optimizer。

## 当前 Delta / 结果 / 下一允许动作（M0 类别1、7统一映射定位）

任务=`M0_BOGT_CLASS17_JOINT_MAPPING_ATTRIBUTION_V1`，唯一 Outer Gate=`M0_BOGT_CLASS17_JOINT_MAPPING_ATTRIBUTION_RESOLVED`。`attempt_01` 保持不可变 `INVALID_HARNESS`，classification=`null`、optimizer=`none`、updates=`0`；类别1、7尚无科学定位结论。

- `preflight_03=READY` 仅绑定旧源码，累计历史资格=`46.326 s`、整卡峰值=`65.54%`，不能用于当前源码。此前已修复 recorder 参数适配、生产生命周期数据形状、观察集行号、逐类别单行梯度、接触图布局、恢复产物校验和历史遥测恢复等问题。
- 本轮确认并修复：从 `preflight_02` 累计值恢复缺失的 elapsed 字段；对 PowerShell 大写与 Python 小写的 SHA-256 作大小写无关的十六进制比较；将历史 `preflight_03` 校验与当前预检预算累计分开；权威解析失败时先写失败阶段和零更新收据。当前只读权威解析报告固定200行拟合集、100行观察集通过，历史46.326秒加未落收据启动检查保守计费15秒=`61.326 s`，既往任务耗时基数=`96.096 s`。
- `preflight_04` 写入一次结果为 `INVALID_HARNESS`，阶段=`authority_resolution_and_runtime_load`，耗时=`0.0292 s`，根因是当时将含15秒新检查计费的总值误用于验证历史 `preflight_03`。失败早于CUDA模型前向、梯度及资源遥测；没有 source manifest、没有global/allocator峰值，也没有科学 attempt。随后修正该比较，CPU authority loader 通过；仍没有可消费的 READY。
- 聚焦pytest=`25 passed`，Windows compileall、聚焦 Ruff、`git diff --check`通过；pytest有既存未知 `cache_dir` 配置警告。protected/test/sealed未读取，`attempt_02`未创建。
- [preflight_04终态](../../outputs/20261001_m0_bogt_class17_joint_mapping_attribution_v1/runs/preflight_04.json)，[authorization_02](../../outputs/20261001_m0_bogt_class17_joint_mapping_attribution_v1/runs/authorization_02.json)，[attempt_01终态](../../outputs/20261001_m0_bogt_class17_joint_mapping_attribution_v1/runs/attempt_01/status.json)，[历史preflight_03](../../outputs/20261001_m0_bogt_class17_joint_mapping_attribution_v1/runs/preflight_03.json)。

原阶段按旧规则停止于 `ROOT_CAUSE_REVIEW_REQUIRED`；`preflight_04` 失败记录不覆盖、不当作 READY。后续普通工程修复遵循当前简化政策，不因次数或摘要变化新增授权编号。本轮审查未授权新科学执行，`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`；不训练、不打开新 attempt，不读取 DEV/protected/test/sealed。

## 历史快照：M0 fixed-small-bank canary（2026-09-30）

### 历史 Delta / 结果（M0 fixed-small-bank joint-fit canary）

任务=`M0_BOGT_SMALL_BANK_JOINT_FIT_CANARY_V1`，唯一 Outer Gate=`M0_BOGT_SMALL_BANK_JOINT_FIT_RESOLVED`。授权03下 `preflight_12=READY`，clean `attempt_02` 已从 accepted step1200 执行恰好1000步到step2200；Adam=`1200→1201→2200`，200个TRAIN拟合行各更新50次。`attempt_01` 原样保留。

- 最终化误报 lineage 的根因：新写入时 `_save_artifact` 返回 `{..., payload: ...}` envelope，已有文件重用时 `_sidecar_payload` 返回内部 payload；消费者把两种形状都直接 `dict()`。已修复并加真实写入/重读回归；聚焦测试=`27 passed`，Windows compileall、Ruff、`git diff --check`通过。
- 原 attempt_02 终态仍为 `INVALID_HARNESS / FINALIZING`；不从终态恢复，不改写为 Gate 结果。修复后仅作只读对账，没有 optimizer 更新、CUDA forward 或新 attempt。训练时使用的 READY 绑定修复前源码，修复后没有新的 READY。
- 保存测量显示拟合集 E_D=`0.355` macro、最低类=`0`；DEV=`0.2220→0.2928`（Δ`+0.0708`，配对 bootstrap 95% CI=`[+0.0620,+0.0798]`）；S1=`0.2320→0.3155`（Δ`+0.0835`，独立 bootstrap 95% CI=`[+0.0625,+0.1050]`）。重建 delta=`−0.003564`；raw CMOS 与光功率检查通过。仅作对账数值，正式 Gate 分类为 null。
- step2200 checkpoint=`1000`步、Adam=`[2200,2200]`、每行更新=`50`；55,000行TRAIN OLS与2,000张S1 raw CMOS均已保存。科学执行整卡峰值=`76.11%`（上限90%）；protected/test/sealed=`0/0/0`。
- [只读对账报告](../../outputs/20260930_m0_bogt_small_bank_joint_fit_canary_v1/runs/finalization_reconciliation_01/reconciliation_report.json)，[清单](../../outputs/20260930_m0_bogt_small_bank_joint_fit_canary_v1/runs/finalization_reconciliation_01/manifest.json)，[根因记录](../../outputs/20260930_m0_bogt_small_bank_joint_fit_canary_v1/runs/root_cause_review_finalization_01.json)，[原 attempt_02 终态](../../outputs/20260930_m0_bogt_small_bank_joint_fit_canary_v1/runs/attempt_02/status.json)。

`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。保持停止；不得追加训练、恢复该已消费 attempt 或据描述性对账自动启动 successor。

## 历史快照：M0 fixed-small-bank canary（预算准入停止时）

任务=`M0_BOGT_SMALL_BANK_JOINT_FIT_CANARY_V1`，Outer Gate=`M0_BOGT_SMALL_BANK_JOINT_FIT_RESOLVED`。预算误判和 DEV JSON 标签类型缺陷均已定向修复，但固定小样本科学训练未开始；`attempt_01` 保持不可变 `INVALID_HARNESS`，`attempt_02` 不存在，科学更新=`0`，没有 Gate 分类。

- `preflight_09=READY`（source semantics=`1b2b57907b07f42141cb8f1ddcff0a8eac13ce6fcb05cedfdd3cdfff9aba209f`，manifest=`576fdb027c70097b835735ae1495fe62e82cc79ea238b4071243bca5c567ebfd`）。旧累计=`476.326 s`，新资格窗口=`162.092/600 s`；整卡峰值=`71.97%`，PyTorch peak allocated/reserved=`2.815/2.917 GB`。旧逻辑把 `163.365 s` 历史 smoke 时长误作必然下界；修复后仅把它用于估计。
- `attempt_01` 已写入冻结 step0 测量，但第一次 optimizer 更新前因 accepted DEV labels 是 JSON 数组而非 Tensor，状态为 `INVALID_HARNESS`（`0` 次更新；Adam 未离开 step1200）。修复后 `preflight_10` 在零 CUDA 阶段暴露路径解析问题，已保留原记录。
- 当前源码绑定 `preflight_11` 在启动 CUDA 前执行硬总预算准入：自首次 CUDA 起=`4,667.048/10,800 s`，剩余=`6,132.952 s`；实测完整 workload 加 20% 余量=`7,070.952 s`，预计超出=`938.000 s`。所以 `preflight_11=PREFLIGHT_BLOCKED(resource)`；这次没有显存超限、没有打开 `attempt_02`、没有使用旧 READY 授权科学执行。preflight_11 source semantics=`2bca9a4c83dc81241f9a174d9b389007adcbc3e23ad9d6b076d1bcd5c227b1a4`，manifest=`d620ed048b59dd45b17d3db414c839938b41949d903b4b16906e214923302d84`。
- 聚焦 pytest=`23 passed`，Windows compileall、Ruff 与 `git diff --check` 通过。证据：[attempt_01 status](../../outputs/20260930_m0_bogt_small_bank_joint_fit_canary_v1/runs/attempt_01/status.json)，[preflight_09 READY](../../outputs/20260930_m0_bogt_small_bank_joint_fit_canary_v1/runs/preflight_09.json)，[preflight_10](../../outputs/20260930_m0_bogt_small_bank_joint_fit_canary_v1/runs/preflight_10.json)，[preflight_11 阻断](../../outputs/20260930_m0_bogt_small_bank_joint_fit_canary_v1/runs/preflight_11.json)。protected/test/sealed=`0/0/0`。

科学合同没有变化；原始三小时截止不重置。继续需要新的明确时间预算授权、当前源码绑定 READY 和新的 clean attempt；不复用或改写 `attempt_01`。`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`；本轮停止。

## 当前 Delta / 结果 / 下一允许动作（canonical gradient augmentation）

唯一 Outer Gate=`M0_BOGT_CANONICAL_GRADIENT_AUGMENTATION_RESOLVED`。clean `attempt_02`=`VALID_COMPLETED / BOGT_CANONICAL_GRADIENT_TRANSFER_SUPPORTED`。规范语义梯度增强相对纯 BOGT step1200 对照提高 DEV 与 TRAIN-neighbor S1 的冻结 E_D macro recall；完整 M0 生成条件仍未满足。

- 执行链复盘确认此前阻断由三个缺口造成：repair-only 收据/CLI 没有 attempt_02 路径；preflight_08 的浮点 κ 检查过严且失败值未先持久化；回归测试没有覆盖 CLI→READY→attempt→终态全链。新增根因审查 06/07、`--run/--resume`、clean attempt_02 身份，以及未 READY 时拒绝数据加载的测试。第一次新增预检仍因 κ 复现检查阻断，原始记录保留；第二次资格复用 preflight_07 已通过的完整基线证据，重新测固定 20 批、真实 3 步、负控、恢复和冻结读出。
- [preflight_09](../../outputs/20260929_m0_bogt_canonical_gradient_augmentation_canary_v1/runs/preflight_09.json)=`READY`，source semantics ID=`afe431ee4ec2ec81d4fe8fb5884b2888e69f6a78e25049bcf8b99031cd2d8e11`，source manifest ID=`e6a9bf6fff0a08d3b674b900f394269457bfdcde1b6e787c3e5fdccaf7b95273`。累计资格时间=`585.81/600 s`，预检峰值整卡=`69.63%`。复测 κ=`0.693675838559751`；科学运行 κ 仍固定=`0.6936758419764932`。
- 正式执行从 accepted step600 独立重建，训练恰好 600 次，Adam=`600→1200`；仅 Encoder 与 Generator 更新。DEV E_D=`0.1318→0.2220`，相对 pure-BOGT step1200=`0.1582`；相对起点 Δ=`+0.0902`，配对 95% CI=`[+0.08099,+0.09781]`。S1 E_D=`0.166→0.232`，Δ=`+0.066`，独立分层 95% CI=`[+0.047,+0.0845]`。DEV 重建 delta=`−0.003138`；raw CMOS/光功率有效。
- 完整 M0 生成数值门槛仍失败：S1 E_D macro=`0.232`，最低每类 recall=`0`。本 Gate 支持 canonical gradient 对独立 E_D 的迁移改善；不等于已解决生成，也不证明架构容量上限。训练总用时=`4559.03 s/4 h`，正式执行峰值全卡=`75.29%/90%`。TRAIN/DEV ledger=`118,200/10,000`；protected/test/sealed=`0/0/0`。
- `attempt_01` 继续保持不可变 `INVALID_HARNESS`，状态哈希=`6708ea49812c43cb238a9aeb11ea5bd32229712e1d69ce9eca5843650ac5483f`。
- [attempt_02 报告](../../outputs/20260929_m0_bogt_canonical_gradient_augmentation_canary_v1/runs/attempt_02/report.json)，SHA-256=`3b1422d8d5ed9d153f60da36547b9e59949d08ffedd3f2db8bf0e59371895a2b`；[终态](../../outputs/20260929_m0_bogt_canonical_gradient_augmentation_canary_v1/runs/attempt_02/status.json)，SHA-256=`13157727c77a1b17921997d251a963691b488850db7cbd02924aac6326f348e4`；[S1 raw CMOS](../../outputs/20260929_m0_bogt_canonical_gradient_augmentation_canary_v1/runs/attempt_02/generated_raw_cmos.pt)；[统一尺度接触图](../../outputs/20260929_m0_bogt_canonical_gradient_augmentation_canary_v1/runs/attempt_02/s1_augmented_step1200_contact_sheet.png)。
- 唯一建议的后续研究域=`BOGT_CANONICAL_GRADIENT_TRANSFER_REPLICATION`，当前没有后续授权，`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。停止；不自动复制实验或延长训练。

## 历史快照：M0 canonical gradient augmentation attempt_01 首次失败

attempt_01 曾在基线初始化处因旧 checkpoint 缺少 `frozen_runtime_state_sha256` 以 `INVALID_HARNESS` 终止，且没有读取数据或执行科学更新。此历史状态、文件和终态均未改写；详见 [attempt 状态](../../outputs/20260929_m0_bogt_canonical_gradient_augmentation_canary_v1/runs/attempt_01/status.json) 与 [旧 preflight_06](../../outputs/20260929_m0_bogt_canonical_gradient_augmentation_canary_v1/runs/preflight_06.json)。

## 前一交接快照（canonical gradient augmentation preflight_03 阻断）

`M0_BOGT_CANONICAL_GRADIENT_AUGMENTATION_CANARY_V1` 的资源修复验证已停止。修复保持原科学目标、20 个固定 TRAIN 校准批次及 90% 全卡显存边界不变；`preflight_02` 和根因审查记录均保留。新的源码绑定 `preflight_03`=`INVALID_HARNESS`，source semantics ID=`a7e0a331ff78a34c1f80b774b160efa6162b57ee3e2b9ecb369e70e162ba5bea`。

- 顺序 E_S/E_I/E_C 梯度等价性、逐批释放测试、24 项聚焦 pytest、Windows 编译、Ruff 和 `git diff --check` 均通过。预检的固定 20 批校准已完成；全卡最高=`67.48%`，PyTorch peak allocated/reserved=`2.541/2.938 GB`；进程级读数不可用。
- 预检在首次 disposable 联合更新的 E_S 顺序梯度阶段触发 harness 断言：trainable parameter `.grad` 非空。该检查未定位 `.grad` 的来源，后续 disposable 更新、恢复检查及测量 smoke 未完成。它不是资源失败或科学结果。
- 新 `attempt_01` 不存在；没有科学 optimizer 更新、科学 checkpoint、S1 生成或 DEV 测量。详细阶段遥测见 [preflight_03_resource_telemetry.jsonl](../../outputs/20260929_m0_bogt_canonical_gradient_augmentation_canary_v1/runs/preflight_03_resource_telemetry.jsonl)，终态见 [preflight_03.json](../../outputs/20260929_m0_bogt_canonical_gradient_augmentation_canary_v1/runs/preflight_03.json)。

本次授权到此消耗完毕。下一动作=`STOP`；不得覆盖或重跑 `preflight_03`，不得创建 attempt。继续需新的明确授权，用只读诊断先定位 trainable `.grad` 的来源。

## 前一交接快照（step1200 只读 BOGT 分析）

唯一 Outer Gate=`M0_BOGT_TUNING_DIRECTION_RESOLVED`；新只读任务 `attempt_02`=`COMPLETED / VALID_COMPLETED`，分类=`NO_SINGLE_WEIGHT_CHANGE_JUSTIFIED`。这不追认原 step600→1200 训练 Gate；原训练 `attempt_02/status.json` 仍是不可变 `INVALID_HARNESS`，状态 SHA-256=`4645e3f01916b3321e4f67bd0eac760434c82dc7c70246789076cd5f2dbd35d0`。

- `preflight_04=READY`，source semantics ID=`949d7d08e86fa6f3ef71354d12b4054abb931057e2b4223f02773061f490971c`；累计预检=`438.68/600 s`，总记录耗时=`2180.12/7200 s`。正式读出整卡峰值=`75.46%`，allocator 峰值 allocated/reserved=`2.812/3.106 GB`。
- 全量重收 step1200 的 `55,000` 行 TRAIN 复 OLS，并按固定 S1 规则生成 `2,000` 张。DEV E_D macro=`0.1318→0.1582`，S1=`0.1355→0.1660`；类别 8 预测占比升至 DEV/S1=`0.7956/0.8015`，最低类 recall=`0`。DEV 重建 delta=`−0.002982`；功率有限，raw CMOS 全部有限、非零且非恒定。原 S1 生成数值门槛未达到。
- 固定的 20 个 TRAIN 批次（cursor `1200–1219`）梯度分析：step1200 Encoder `αg_dist/βg_rec/γg_op` 中位数=`0.001499/0.001016/0.001059`；Generator `L_dist` 是结构零，`βg_rec/γg_op`=`0.001419/0.000168`。候选 A/B/C=`4/20、0/20、0/20`，均未达到 `16/20` 一致性标准，故只读结论为 `NONE`，不建议单因素调权。
- optimizer/update/E_D gradient=`0/0/0`；step600、step1200 参数未变，`.grad` 字段为空。TRAIN/DEV=`55,400/10,000`；protected/test/sealed=`0/0/0`。新分析没有 checkpoint，也不改变原训练状态。
- [报告](../../outputs/20260929_m0_bogt_step1200_evidence_gradient_balance_v1/runs/attempt_02/report.json)，SHA-256=`067ffca250a5469447c5266f46a26e917f7d221b9e305c98bb792e7f839391aa`；[逐批梯度结果](../../outputs/20260929_m0_bogt_step1200_evidence_gradient_balance_v1/runs/attempt_02/gradient_analysis.json)；[raw CMOS](../../outputs/20260929_m0_bogt_step1200_evidence_gradient_balance_v1/runs/attempt_02/generated_raw_cmos.pt)；[接触图](../../outputs/20260929_m0_bogt_step1200_evidence_gradient_balance_v1/runs/attempt_02/step1200_s1_contact_sheet.png)。

后续调权假设=`NONE`，`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。停止；不改 α/β/γ、不续训、不启动 successor。

## 历史快照：M0 原三项 BOGT 联合 canary 的首次 step1200 收尾恢复

以下内容保留 2026-09-28 首次最终化恢复的历史状态；上方新任务后来独立完成了只读读出，没有修改该训练 attempt 或追认其 Gate。

唯一 Outer Gate=`M0_BOGT_JOINT_GENERATION_CANARY_RESOLVED`。accepted step600→step1200 恰好完成 600 次联合更新，目标只有 `αL_dist + βL_rec + γL_op`；step1200 checkpoint 和最新恢复点对齐，Adam 内部步数=`1200/1200`。step800/1000/1200 TRAIN minibatch 损失见 [CURRENT_STATUS](../CURRENT_STATUS.md)，数值非单调，只是描述性记录。

- 训练 attempt_02 原终态仍为不可变 `INVALID_HARNESS`：step1200 checkpoint 写好后，收尾 heartbeat 使用非法 phase `POST_TRAIN_OLS_COLLECTION`。后续收尾恢复无 optimizer、无训练更新；source-bound `finalization_preflight_01=READY`，forward-only smoke 和数据/端点身份检查通过。
- 收尾阶段在 55,000 行 TRAIN OLS bank 收集时，检测到全卡使用比例=`0.9014`，超过 0.90 上限并停止。此前已落盘并核实连续 22,000 行（22×1,000）；bank 不完整，未用于生成。错误抛出早于阶段遥测写盘，所以该时点的确切整卡字节与 PyTorch allocator 峰值不可恢复，不作来源推测。
- 没有 step1200 S1、raw CMOS、contact sheet、最终 DEV E_D 或 Gate report。科学 Gate 无分类；工程最终化状态=`INTERRUPTED`。`attempt_02/status.json` 哈希保持不变，`attempt_01` 也保持原样。只读停止记录：[resource_stop_observation.json](../../outputs/20260928_m0_bogt_joint_generation_canary_v1/runs/attempt_02/finalization_recovery_01/resource_stop_observation.json)；[finalization_preflight_01](../../outputs/20260928_m0_bogt_joint_generation_canary_v1/runs/finalization_preflight_01.json)。
- protected/official test/sealed=`0/0/0`。当前动作=`STOP`，`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`；不要据此判训练科学失败，也不自动重新收尾或续训。任何新执行需先解决资源观测/容量条件并获得新的明确授权，原科学工作量与门槛不应静默更改。

## 前序 Delta / 阻塞 / 下一允许动作（M0 canonical semantic canary）

唯一 Outer Gate=`M0_CANONICAL_SEMANTIC_RENDERABILITY_CANARY_RESOLVED`；source-bound `preflight_12=READY` 后，clean `attempt_02` 有效完成，classification=`CANONICAL_SEMANTIC_LOCAL_GAIN_WITHOUT_INDEPENDENT_DATASET_TRANSFER`。固定 TRAIN fit-bank 的 E_S/E_I/E_C 特征损失均下降；三项描述性等权均值从 `0.928731` 降至 `0.837968`（该均值不是实际优化标量）。100 行 observation 的 E_D macro=`0.15→0.22`。Full DEV E_D macro=`0.1318→0.1594`，Δ=`+0.0276`，999 次类别分层配对 bootstrap 95% CI=`[+0.0214,+0.0332]`。因增量未达预注册 `+0.05` practical-transfer 阈值，分类为局部改善而非独立数据集转移。

- 600 秒累计预检限额来自 `long-training-preflight.md`：`min(10 min, 2 h × 10%)=600 s`。本次 source-bound 收据累计=`560.793 s`；最大整卡显存=`4.834/6.000 GiB (80.6%)`。step600→新训练 runtime 隔离成立；预检状态全部丢弃，scientific Adam fresh-zero，审计=`0→1→300`，恰好 300 次 Generator 更新。
- DEV 重建 delta=`−0.0009222`（限值`+0.005`），光功率有限，raw CMOS 有效；仅更新 `generator.phase`，E_D 仅确认。TRAIN fit/observation=`200/100`，DEV=`5,000`（step0/300），protected/test/sealed=`0/0/0`。attempt_01 仍为不可变 `INVALID_HARNESS`。
- [attempt_02 报告](../../outputs/20260928_m0_canonical_semantic_renderability_canary_v1/runs/attempt_02/report.json)，SHA-256=`c068133313617a8e8bd69be7a1c0c5c50d0b393ce2262ab203dc30678aa9bec1`；[终态](../../outputs/20260928_m0_canonical_semantic_renderability_canary_v1/runs/attempt_02/status.json)；[source-bound READY](../../outputs/20260928_m0_canonical_semantic_renderability_canary_v1/runs/preflight_12.json)，source semantics ID=`0eae5b50e99dcf474187a1b9b465b4756c8d1cfbb228c055da943f85807e51f2`。

唯一建议的后续研究域=`CANONICAL_TRANSFER_FAILURE_ATTRIBUTION`；这是研究方向，不是授权。`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。本任务已停止，不自动创建后继实验。

## 前序交接（M0 AutoDL attempt_01 只读证据恢复 Case D）

以下是 canonical clean `attempt_02` 完成前的状态快照；关于本地 attempt_02 尚不存在的描述仅反映当时状态。

## 历史 Delta / 阻塞 / 下一允许动作（AutoDL evidence recovery）

唯一 Outer Gate=`M0_AUTODL_ATTEMPT_01_EVIDENCE_IDENTITY_RESOLVED`；只读核查已完成，分类=`CLOUD_RUN_EVIDENCE_PARTIALLY_RECOVERED_BUT_CAUSAL_IDENTITY_UNRESOLVED`（Case D）。详见[证据恢复报告](../../outputs/20260928_m0_autodl_attempt_01_evidence_identity_recovery_v1/runs/recovery_01/report.json)。

- 本地 AutoDL 候选包的 2,000 个保存视图在同一冻结 E_D 上复核一致：macro=`0.8545`、最低类=`0.485`、类别 8 预测占比=`0.139`。这确认保存视图评分，不绑定训练轨迹。
- 缺少最终模型、`generated_ols.pt`、训练进度/云端日志、optimizer 状态、云端源码副本及 provider job/session 身份；只保存 50 张 raw CMOS 示例。AutoDL 页面读取连续两次超时，当前无法核对远端目录。候选与用户所说大型 cloud run 的对应关系未确认。
- 候选源代码声明的是 seed-1234 新模型、Encoder+Generator 600 步、重建 MSE + 单一分类器 CE、类条件高斯采样；其起点、目标、可训练范围、输入与光学配置均不同于本地 accepted step600 的三评价器因果问题。
- 本地 canonical canary 仍 `INVALID_HARNESS`；有效 V2 定位仍 `NEITHER_BOUNDED_UPPER_BOUND_REACHED`。两者均不支持架构不可能语义生成的结论。

下一步只需用户提供原 AutoDL run/job ID 或该次运行的只读导出目录/对象存储位置，以便继续恢复既有证据。`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`；不训练、不续训、不创建 local attempt_02、不重跑 cloud run 或 promotion。

## 前序 Delta / 阻塞 / 下一允许动作（M0 evidence reconciliation）

本节记录的是 clean canonical `attempt_02` 执行前的复核状态；其中关于 `attempt_02` 未创建及仅进行 AutoDL 证据恢复的描述已由上方当前 Gate 结果取代。

唯一 Outer Gate=`M0_CANONICAL_SEMANTIC_TRAINING_STATE_RESOLVED`，只读复核结果=`VALID_COMPLETED_EVIDENCE_RECONCILIATION`；最终分类=`CLOUD_RUN_INVALID_OR_INCOMPLETE_FOR_SCIENTIFIC_USE`。详见[完整复核报告](../../outputs/20260928_m0_canonical_semantic_training_evidence_reconciliation_v1/runs/reconciliation_01/report.json)。

- 本地 canonical canary `attempt_01` 仍为 `INVALID_HARNESS`：训练循环计数为 300，但 Adam 状态从 4 走到 304；收尾 heartbeat 又使用了非法 `COMPLETE` phase。科学分类为 null。报告中的 `0.1318→0.1596` 及 bootstrap 区间只能作为非科学调试信息。无 `attempt_02`，修复后的源码没有新的 READY。
- 唯一匹配的 AutoDL 本地包是 400×400、2+4 相位层的 600 步工程基线，从 seed 1234 新建模型，训练 Encoder 与 Generator，目标为重建 MSE + 单个冻结分类器 CE，生成端用 class-conditional diagonal complex Gaussian。它不是本地 step600 固定 OLS / Generator-only / `E_S/E_I/E_C` canary。
- 云端报告生成 E_D macro=`0.8545`、每类=`[0.985,0.485,0.850,0.925,0.990,0.780,0.975,0.930,0.950,0.675]`；2,000 个保存的 evaluator views 上独立复算得到完全相同的混淆矩阵，E_D 无梯度。真实 DEV 报告值为 `0.9384`，没有同模型 step0 对照。
- 同步包缺失报告所称 `model_step_000600.pt`、`generated_ols.pt`、每 25 步日志和机器/GPU/session 身份；只有 50 张 raw CMOS 示例而非全量 2,000 张。它的报告标记 `diagnostic-only`，所以这些数字可以作为探索性读数，不能确认训练终点和轨迹，也不能回答原 M0 的目标因果问题。缺文件不等于云端运行失败，可能是同步不完整。
- accepted step600 的 DEV/S1 E_D macro=`0.1318/0.1355`；先前有效的 V2 像素目标上界测量为 `NEITHER_BOUNDED_UPPER_BOUND_REACHED`。综合现有证据仍不能断言当前 M0 架构不可能语义生成。

`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。建议的唯一后续任务：`M0_AUTODL_ATTEMPT_01_EVIDENCE_RECOVERY_AND_IDENTITY_CONFIRMATION_V1`，仅找回并核验已存在的云端产物和身份，不续训。原 fixed-step600 三评价器受控比较的干净 `attempt_02` 在问题仍决策关键时仍有科学必要，但当前未授权、未创建。

## 前一 Gate 结果（M0 OLS 尺度与梯度修复）

Delta：`M0_BOGT_OLS_SCALE_GRADIENT_RESCUE_RESOLVED` 的唯一 attempt_01 已完成，终态=`VALID_FAILED`，分类=`SATURATION_PERSISTS_WITHIN_BOUND`。在 step600 冻结光学模型上，对固定 200 张 TRAIN 图像的真实 OLS 输入执行 6,000 步；每行独立 Adam 恰好更新 300 次，单位化输入偏移受半径 1 约束。`preflight_02=READY`，无工程完整性失败。

- 前景 MSE `0.398954→0.057281`（降 `85.64%`）；背景 MSE `0.008762→0.012235`（升 `39.63%`）。E_D macro=`0.82`，逐类 recall=`[1.00,0.65,0.95,0.90,0.70,0.70,0.95,0.70,1.00,0.65]`。
- raw CMOS 有限、非零且非恒定；`2.6195%` 像素仍超过 1，高于 `1%` 上限。相对 OLS 偏移最大值=`1.00000016`，处于投影的浮点容差内。按照预注册规则，饱和条件决定 Gate 分类；不得把 E_D 和前景改善当作 Gate 通过。
- [报告](../../outputs/20260928_m0_bogt_ols_scale_gradient_rescue_v1/runs/attempt_01/report.json)、[raw CMOS/OLS/逐样本张量](../../outputs/20260928_m0_bogt_ols_scale_gradient_rescue_v1/runs/attempt_01/scale_rescue_measurements.pt)、[起点接触图](../../outputs/20260928_m0_bogt_ols_scale_gradient_rescue_v1/runs/attempt_01/start_contact_sheet.png)、[终点接触图](../../outputs/20260928_m0_bogt_ols_scale_gradient_rescue_v1/runs/attempt_01/end_contact_sheet.png)。登记的 SHA-256 均复核一致。
- 总耗时=`850.10 s`（预检 `12.58 s`、正式执行 `836.11 s`），峰值整卡使用=`46.71%`；TRAIN 读取 200 行，DEV/protected/test/sealed=`0`。聚焦 pytest=`11 passed`，Windows compile、Ruff、diff check 通过。
- 定向诊断支持“尺度过大的无界 OLS 更新会导致强度爆炸和截断梯度失效”；本次有界尺度修复显著改善前景和 E_D，但没有消除末端饱和、背景误差也上升。故尺度/截断是重要因素，尚未证明唯一根因；不能据此判断物理容量不足。
- `preflight_01` 的 `KeyError: prior_report` 属于 attempt 前 harness 错误，已用一次定向修复与 `preflight_02 READY` 收敛；原记录保留，不是科学负结果。原 M0 V2 两臂结果继续作为历史对照，不改写。

`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。本结果不授权长训练、扩展 OLS 输入范围、使用优化系数作为采样器、改截断/标定或修改架构。下一动作应先设计一个独立、边界明确的诊断 Gate，定位剩余 `2.6195%` 饱和与背景误差上升的来源，再单独授权执行。

## 前序 V2 两臂可呈现性诊断（历史）

此前 V2 OLS 臂相对 OLS 偏移均值约 `600.8` 倍，raw CMOS 均值约 `21,858.74`，接近全白饱和；该现象促成本次受限尺度测试。V2 Gate 仍为其原有的 `VALID_COMPLETED_DIAGNOSTIC / NEITHER_BOUNDED_UPPER_BOUND_REACHED`，未被本次报告覆盖。详见[原报告](../../outputs/20260927_m0_bogt_renderability_localization_v2_heartbeat_recovery_v1/runs/attempt_01/report.json)。

- 执行链根因按顺序为：`preflight_07` 全 TRAIN 编码令整卡占用达到 `6.44/6.44 GB`、allocator reserved 峰值 `5.27 GB`，触发 90% 门；`preflight_08` 收据消费者引用了未登记的 Generator/OLS 三步更新键；`preflight_09` 的 READY 没覆盖正式执行 heartbeat，旧 attempt 在首次更新前写入非法 phase 并终止。旧状态和收据均保留，不能复用。
- [heartbeat 根因审查](../../outputs/20260927_m0_bogt_renderability_localization_v2/runs/root_cause_review_04.json) 后，新授权使用独立 run。其 [preflight_01](../../outputs/20260927_m0_bogt_renderability_localization_v2_heartbeat_recovery_v1/runs/preflight_01.json) 绑定 source semantics=`f509643902cf872c4a25cfbb6014a6d6ff39b00b507177282b0f678b898440a9`，状态=`READY`，耗时=`21.98 s`，累计预检=`163.25 s`，预测总时长=`1,206.38 s`。各阶段同时记录整卡使用与 PyTorch allocator；最大整卡占用=`3.313 GB/6.442 GB (51.43%)`，allocator 峰值 allocated/reserved=`1.838/2.147 GB`。三步更新、有限差分、冻结/负控、跨 OLS chunk 边界的恢复等价性、heartbeat 与完整测量和写盘探针通过。
- 新 run 的唯一 [attempt_01](../../outputs/20260927_m0_bogt_renderability_localization_v2_heartbeat_recovery_v1/runs/attempt_01/status.json) 已完整终结：Generator 恰好 `300` 步，OLS `6,000` 步且每行独立更新 `300` 次；最终状态=`COMPLETED / NEITHER_BOUNDED_UPPER_BOUND_REACHED`，实际总耗时=`1,034.37 s/4 h`。产物哈希已核对。旧 V2 尝试仍是 `INVALID_HARNESS`；新结果只完成原诊断，不授权主线长训练或任何自动后续动作。

## 前序有效 Gate（M0 canonical image rescue；VALID_FAILED）

Delta：`M0_BOGT_CANONICAL_IMAGE_RESCUE_RESOLVED` 的 attempt_02 从 accepted step600 完成 300 步并通过完整性检查，结果=`VALID_FAILED`。DEV E_D macro=`0.1318→0.1428`（Δ`+0.0110`，配对 95% CI `[+0.0060,+0.0164]`）；S1=`0.1355→0.1440`（Δ`+0.0085`，独立 CI `[−0.0075,+0.0240]`）。两者均未达到 `+0.05`。非 8→预测 8 比例 DEV/S1 分别从 `0.7667/0.7772` 增至 `0.7987/0.8111`；类别 8 预测占比达 `0.8050/0.8165`，集中误判未缓解。该结果否定本次短程目标达到门槛，不证明架构容量不足。

- DEV reconstruction delta=`−0.0011382`，2,000 张新 raw CMOS 全部有限、非恒定、非零，功率有限，冻结/数据角色通过；protected/test/sealed reads=`0/0/0`。55,000 行当前 TRAIN OLS 支撑固定同类 16-NN S1，每类 200 张，共 2,000 张。
- TRAIN-only 全局仿射定位使用 gain=`1.44472`、offset=`0.01891`；校正后 step600 DEV/S1 E_D macro=`0.1250/0.1300`，未满足原 `≥0.50`、每类 `≥0.10` 生成目标，简单亮度标定不能解释或消除该失败。raw CMOS 与系统标定未修改。
- step900 DEV/S1 macro=`0.1428/0.1440`、最低类 recall=`0/0`；原完整生成标准 `macro≥0.50` 且每类 `≥0.10` 仍未达到。
- [Gate 报告](../../outputs/20260926_m0_bogt_canonical_image_rescue_v1/runs/attempt_02/report.json)，SHA=`d3d6f7eeb6ff9e30ac4ca70c22da8fcd967916492ead8fb917ddd24cdbc672ba`；[step900 checkpoint](../../outputs/20260926_m0_bogt_canonical_image_rescue_v1/runs/attempt_02/scientific_step_000900.pt)；[raw CMOS tensor](../../outputs/20260926_m0_bogt_canonical_image_rescue_v1/runs/attempt_02/generated_raw_cmos.pt)；[S1 接触图](../../outputs/20260926_m0_bogt_canonical_image_rescue_v1/runs/attempt_02/s1_contact_sheet_10_per_class.png)；[DEV 对照图](../../outputs/20260926_m0_bogt_canonical_image_rescue_v1/runs/attempt_02/fixed_dev_real_contact_sheet.png)。图片是 TRAIN 邻域采样结果。
- source-bound [preflight_05 READY](../../outputs/20260926_m0_bogt_canonical_image_rescue_v1/runs/preflight_05.json)，source semantics ID=`7f443f6039dcfd9f25a241cf9ec735e5e30243daa7f8c88e28248089d023f076`，估算=`1503.0 s`、峰值=`67.3%`；attempt_02 总耗时=`908.0 s`、峰值=`82.4%`。聚焦 pytest=`11 passed`，Windows compile、Ruff 和 diff check 均通过。
- 原 [attempt_01 状态](../../outputs/20260926_m0_bogt_canonical_image_rescue_v1/runs/attempt_01/status.json) 保持 `INVALID_HARNESS`：第 601 步发生 inference tensor/backward 错误，没有模型 checkpoint 或完整报告。已把定位前向从 inference mode 改为 no-grad，并使预检按真实执行顺序；attempt_02 从 accepted step600 全新加载。attempt_01 的终态和文件未覆盖。
- `CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。不自动延长训练、改采样器、改架构或标定、不重开 N3、不读 protected/test/sealed、不 promotion；任何下一项工作需独立研究决策。

## 前序类别 8 定位状态（只读；2026-09-26）

Delta：唯一 Outer Gate=`M0_BOGT_CLASS8_ATTRACTION_LOCALIZATION_RESOLVED` 已完成，分类=`FEATURE_ONLY_CLASS8_ALIGNMENT`。在固定 TRAIN 类中心下，step900 DEV/S1 中 E_D 判为 8 的非 8 类样本几乎都最近于 E_D 的 8 类特征中心，却几乎没有最近于像素 8 类中心。此结果描述固定距离测量，不证明因果机制。

- step900 DEV 的 3,434 个非 8→预测 8 样本：特征中心最近为 8 的比例=`0.996214`（95% CI=`[0.994176,0.997962]`），像素中心=`0.003203`（`[0.001456,0.005242]`）。step900 S1 的 1,379 个此类样本：特征=`0.999275`（`[0.997825,1.000000]`），像素=`0.001450`（`[0,0.003626]`）。两个群体都满足预设 feature lower >0.50、pixel upper <0.50。
- E_D 类别 8 预测占比 DEV/S1=`0.7704/0.7740`；非 8 请求类被判为 8 的比例=`0.7631/0.7661`。完整每类 recall、混淆矩阵和逐样本值见报告。DEV step600/900 是同一行配对；两组 S1 来自不同保存 bank，只作独立比较。
- E_D 原始 49,500 行 TRAIN fit 仅用于构造像素和 256D 特征均值；5,500 行资格集混淆矩阵匹配既有 authority，macro recall=`0.9811`。仅冻结前向、无 optimizer/backward/参数更新；protected/test/sealed reads=`0/0/0`。测量=`118.9 s`、预检=`10.0 s`、峰值显存=`81.8%`。
- source-bound [preflight_06 READY](../../outputs/20260926_m0_bogt_class8_attraction_localization_v1/runs/preflight_06.json)，source semantics ID=`3893454ae7e02a7ff3f01a54dcb634f66b23f925952792c0fad3e1cb5413c524`，估计总耗时=`423.4 s`。此前五份失败预检均未打开 attempt，记录保留。
- [定位报告](../../outputs/20260926_m0_bogt_class8_attraction_localization_v1/runs/attempt_01/report.json)，SHA=`4ca77be5fb382bab6c7a50497fa77b450bdc93fd6a1c50fe55e2ac3a5b012317`；[精简逐样本向量](../../outputs/20260926_m0_bogt_class8_attraction_localization_v1/runs/attempt_01/localization_vectors.pt)，SHA=`28a067d336f6710a642cd0852ed8c50de72b294832e0e77b92fcecb493f320fd`。该 run 未复制大型 raw CMOS tensor。
- [执行器](../../scripts/research/diagnose_lightgen_2d_m0_bogt_class8_attraction_localization_v1.py)和[聚焦测试](../../tests/test_m0_bogt_class8_attraction_localization.py)：focused pytest=`11 passed`，Windows compile、Ruff、`git diff --check` 通过；测试覆盖 DEV 固定行配对、S1 独立标签 bank、只读 evaluator 前向和终态写入。
- 原 step600→900 训练 canary 的 `attempt_01` 继续保持 `INVALID_HARNESS`；本只读 Gate 不追认其训练结果。`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`；不自动续训、改采样器、重开 N3、读取 protected/test/sealed 或 promotion。后续干预需新的研究问题和执行授权。

## 前序 M0 step900 方向证据独立复核

### 独立复核详情

Delta：对 M0 step600 与 step900 已保存 checkpoint、固定 DEV 5,000 行和原有 S1 2,000 个复输入完成只读独立重放。唯一审计 Gate=`M0_BOGT_CANONICAL_DIRECTION_EVIDENCE_AUDIT_RESOLVED`，结果=`COMPLETED / EXISTING_DIRECTIONAL_EVIDENCE_REPRODUCED`；49 项一致性检查全部通过。它只确认短程方向数值可重现，不改变原训练 canary 的 `INVALID_HARNESS` 状态，也不追认其 Gate。

- DEV E_D macro recall=`0.1318→0.1556`，配对分层 bootstrap Δ=`+0.0238`、95% CI=`[+0.018395,+0.0298]`；step600 DEV 预测逐行匹配已存向量。S1 E_D macro=`0.1355→0.1600`，独立分层 bootstrap Δ=`+0.0245`、95% CI=`[+0.0080,+0.0410]`；预测逐行一致，raw CMOS 最大绝对差=`0`。DEV reconstruction delta=`−0.0005659`，2,000/2,000 S1 raw CMOS 有限且非恒定、零图=`0`，功率有限；S1 class-8 mass=`0.774`，类别 recall 仍不满足完整生成目标。
- 审计仅做冻结前向和统计，无 optimizer/backward/update；DEV/S1 角色核对通过，protected/test/sealed reads=`0/0/0`。source-bound `preflight_02=READY`，source semantics ID=`b260be796bec2f05136a8465c6bdc264e12c7f78ae17dd5b411db4d40a30e4cc`；含已记录的前次失败和本次预检，完整执行估算=`1083.8 s`。报告执行计时=`391.6 s`（含前次 190.2 s），本次预检=`9.2 s`；观测显存峰值=`88.6%`。
- [独立复核报告](../../outputs/20260926_m0_bogt_canonical_direction_evidence_audit_v1/runs/attempt_01/recovery_01/report.json)，SHA=`be48e431dbfcf9289d9f1235ed6fa6272fa035c6605d0f321d94ada391c7ce92`；[预测向量](../../outputs/20260926_m0_bogt_canonical_direction_evidence_audit_v1/runs/attempt_01/recovery_01/replayed_prediction_vectors.pt)，SHA=`1e367c437fc8fa31138089592f572e59e41b99dfca0fdcf51854982a8cc068b4`；[READY 收据](../../outputs/20260926_m0_bogt_canonical_direction_evidence_audit_v1/runs/preflight_02.json)。首次审计 `attempt_01` 因错误的功率汇总函数保留为 `INVALID_HARNESS`；显式 `recovery_01` 绑定其状态哈希并完成复核。
- 原训练 canary [report](../../outputs/20260926_m0_bogt_active_canonical_semantic_direction_canary_v1/runs/attempt_01/report.json)虽记录短程 Gate 数值通过，其 [status](../../outputs/20260926_m0_bogt_active_canonical_semantic_direction_canary_v1/runs/attempt_01/status.json) 与 [failure report](../../outputs/20260926_m0_bogt_active_canonical_semantic_direction_canary_v1/runs/attempt_01/failure_report.json) 仍为 `INVALID_HARNESS`，未触碰。审计结果不授权续训、attempt_02 训练、采样器更改、重开 N3 或 promotion；`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。
- 限制：先前训练 `preflight_03` 的数值 trace 差异幅度未记录；本复核确认 DEV 与保存 S1 的数值结果，不证明三步训练恢复逐位一致性。

## 前序训练 canary 状态（2026-09-26；M0 canonical direction attempt INVALID_HARNESS）

Delta：step600→900 恰好完成 300 次更新；`report.json` 内部记录唯一 Outer Gate=`M0_BOGT_CANONICAL_DIRECTION_RESOLVED` 通过，但 attempt 的最终 `status.json`=`INVALID_HARNESS`。执行器在报告写入前后各调用了一次 `control.complete()`，第二次调用报 `RunControlError: only an active task can complete`。测量报告、检查点和样本均完整落盘，内部 artifact hashes 已核对；由于 lifecycle 终态无效，不把该报告作为已接受的 Gate 结果。

- DEV E_D macro recall=`0.1318→0.1556`，Δ=`+0.0238`，配对分层 95% CI=`[+0.0184,+0.0298]`；S1=`0.1355→0.1600`，Δ=`+0.0245`，独立分层 CI=`[+0.0080,+0.0410]`。两项满足短程方向门槛。step900 S1 class-8 prediction share 仍为 `0.774`；S1 per-class recall=`[0.210,0,0.210,0.050,0.140,0.135,0.010,0,0.845,0]`，远低于完整生成目标。
- DEV reconstruction delta=`−0.0005659`；2,000/2,000 raw CMOS finite、nonconstant、nonzero；power finite。BOGT Encoder 梯度非零，Encoder/Generator 均更新，冻结和数据 firewall 通过，protected/test/sealed reads=`0/0/0`。实际耗时=`921.4 s`，峰值显存=`82.6%`，在预算内。
- [报告](../../outputs/20260926_m0_bogt_active_canonical_semantic_direction_canary_v1/runs/attempt_01/report.json)，SHA=`6eb87e5ce185696d7212818a8a7f5d52f19b820f0c1dfad072a46cb4eb79d59c`；[raw CMOS tensor](../../outputs/20260926_m0_bogt_active_canonical_semantic_direction_canary_v1/runs/attempt_01/generated_raw_cmos.pt)；[S1 接触图](../../outputs/20260926_m0_bogt_active_canonical_semantic_direction_canary_v1/runs/attempt_01/s1_contact_sheet_10_per_class.png)；[attempt status](../../outputs/20260926_m0_bogt_active_canonical_semantic_direction_canary_v1/runs/attempt_01/status.json)；[failure report](../../outputs/20260926_m0_bogt_active_canonical_semantic_direction_canary_v1/runs/attempt_01/failure_report.json)。failure report 的 `automatic_recovery_permitted=false`；不续跑原训练 canary、不改写其产物或终态、不创建训练 `attempt_02`。上方独立审计另行保留复核 provenance。
- 已修复执行器的重复终态调用，并新增回归测试；Windows compile、Ruff、focused pytest=`8 passed`。修复晚于本次 attempt，不追认旧 attempt 状态；若未来重开，必须重新授权并基于修复后源码重新预检。

## 历史预检记录（2026-09-26；M0 canonical semantic direction）

最初直接启动 Python 时未把受支持环境的 `bin` 加入 DLL 搜索路径；匹配的 CUDA 12.1 NVRTC 组件实际存在。仅给实验进程临时补入该目录后，CUDA 复数运算正常。`preflight_02` 与 `preflight_03` 分别暴露 RNG 起点恢复和浮点 trace 比较问题；修复后 `preflight_04=READY`，source semantics ID=`2782cf2349c3c9f5ac620695cb2da350ba4e824bf9371d5f50f3481b21b7906b`，预计总执行=`1411.1 s`、预检峰值显存=`69.1%`。

## 历史状态快照（2026-09-26；M0 CMOS decomposition report 已落盘，finalization 需根因审查）

Delta：对 accepted M0 step600 完成只读 CMOS class-code decomposition。`report.json` 的 Outer Gate=`M0_CMOS_SEMANTIC_REALIZATION_DECOMPOSED`，result=`VALID_COMPLETED_DIAGNOSTIC`，classification=`CMOS_LINEAR_CLASS_CODE_PRESERVED_BUT_CANONICAL_REALIZATION_FAILED`。TRAIN-only 固定线性 probe 的 real DEV macro recall=`0.9690`（95% CI=`[0.9642,0.973605]`，min-class=`0.956`），既有 S1 raw CMOS=`0.9970`（`[0.9945,0.9990]`，min-class=`0.990`）；两者超过定位标准 `macro≥0.50 / min-class≥0.10`。这支持 step600 raw CMOS 保留稳定、线性可解码的类条件编码，但独立 E_D 仍失败，不能声称信息论层面的结论或独立先验新颖性。

- 数据/评价：仅在 canonical TRAIN 55,000 行拟合 CMOS probe；DEV=5,000、既有 S1=2,000 只作评价。没有精确绑定的 step600 TRAIN raw-CMOS artifact，故一次冻结前向以小批 raw 输出逐批执行注册的 CPU 28×28 pooling；不保存全量约 8.8 GB raw tensor。E_D 对 real/S1 的 macro=`0.1318/0.1355`、逐样本预测 SHA、混淆矩阵与预测直方图均精确复现前序接受报告。可选训练语义 evaluator 仅描述性评估：DEV/S1 macro=`0.2084/0.2130`，不参与分类。

- source-bound `preflight_recheck_01=READY`，source semantics ID=`da0a0278328197008fd5b8bc419552a96fa519e3022cf917c4f02e47a2fa68f0`；100 行/类真实冻结 smoke=`PASS`；估计峰值 GPU memory=`56.96%`，估计总执行=`3235.86 s`，实际包含预检耗时约=`387.09 s`。TRAIN raw 前向 55,000 行、DEV 5,000 行、S1 raw 2,000 行复用；光学参数更新/optimizer/checkpoint=`0/0/0`，protected/test/sealed reads=`0/0/0`。一个 LBFGS optimizer 只用于 TRAIN-only diagnostic probe。
- 前序 authority：M0 report SHA=`e9814c90bc6600f7cccda1f7ec4e54d560c22d7d67553db37b0e26a7516ab8d4`；step600 checkpoint=`ee8616e6c3962d305c8571dbd34de20ae1254be76bc88a963f2ebf57e8f3074e`；TRAIN OLS=`a7e627a6a1296152d182cb7a0b19c659c288dcb6127bedf34559ee040863be1a`；S1 OLS/raw CMOS=`c0f5239e7d0190bd16a77bb63c724219184e8eb3c81ae556b1a508bb2f261a5f` / `50fb01042081313184fa225dcc5cadb99393473f8200c9efe5d148cbda91477b`；冻结 E_D checkpoint=`b8ece3e3bacedd5b5bec63073bee02ce7489161030b835740c681a1ebd3f9ff9`。完整 identity/authority 表见新报告。
- [分解报告](../../outputs/20260926_m0_bogt_cmos_semantic_realization_decomposition_v1/runs/attempt_01/report.json)，SHA=`0d2596734bfc7d4fbb8b54d159aa592916d07db7fabc50335d45d559c7edbc54`；[TRAIN-only probe](../../outputs/20260926_m0_bogt_cmos_semantic_realization_decomposition_v1/runs/attempt_01/cmos_probe.pt)，SHA=`5a40b66ef0e93970b839749636626d601eb7ac391fbc339d25bc0b76dc159656`；[READY 重检](../../outputs/20260926_m0_bogt_cmos_semantic_realization_decomposition_v1/runs/attempt_01/preflight_recheck_01.json)。报告中含逐类 recall、混淆矩阵、预测直方图和 bootstrap CI。
- lifecycle 异常：`status.json=INVALID_HARNESS`，错误为 `TypeError: atomic_write_json() got an unexpected keyword argument 'artifact_kind'`。该异常发生在 report/probe 已写入、E_D fidelity 及冻结/RNG 检查完成后的终态写入。相同错误签名在一次 targeted 修复后再次出现；科学 report 可读且内部 Gate 已解析，但 lifecycle status 仍不完整。记录 `ROOT_CAUSE_REVIEW_REQUIRED`，保留现有状态，不再自动修状态或重跑。
- 分类所暗示的后续研究域为 `CANONICAL_SEMANTIC_REALIZATION_OBJECTIVE`，**不是执行授权**。`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`；不启动训练、不改 S1、不重开 N3、不读取 protected/test/sealed、不 promotion。

## 历史状态快照（2026-09-26；M0 BOGT generation-failure localization）

上一只读 Outer Gate=`M0_BOGT_GENERATIVE_FAILURE_LOCALIZED`，结果=`VALID_COMPLETED_DIAGNOSTIC`，classification=`OLS_TO_CMOS_SEMANTIC_REALIZATION_BOTTLENECK_SUPPORTED`。DEV real OLS macro recall=`0.9214`（95% CI=`[0.9138,0.9286]`），S1 OLS=`0.9600`（`[0.9510,0.9685]`）；DEV real CMOS/E_D=`0.1318`（`[0.1240,0.139005]`），S1 raw CMOS/E_D=`0.1355`（`[0.1235,0.1480]`）。这项报告提供本次 E_D 复现基准。
- [前序诊断报告](../../outputs/20260926_m0_bogt_generative_failure_localization_v1/runs/attempt_01/report.json)，SHA=`3f3facaa5e963fac06807b36da760330ff8cf3d13f7b8a831db75972bc09e5cc`。此前用户授权的 diagnostic 与其 `preflight/status` 均保留。

## 历史状态快照（2026-09-26；M0 BOGT semantic-generation canary VALID_FAILED）

Delta：M0 从 accepted Chen-inspired BOGT step300 出发，完成恰好 300 步至 step600；同类 16NN S1 只从当前模型 TRAIN 复 OLS bank 生成每类 200 个、共 2000 个复输入，并经固定 `W_OLS → Generator → raw CMOS` 路径前向。唯一 Outer Gate=`M0_BOGT_SEMANTIC_GENERATION_CANARY_RESOLVED`，有效结果=`VALID_FAILED`。

判定：2000/2000 raw CMOS 输出有限、非恒定、非零；光功率有限；step600 DEV `L_rec` delta=`-0.0019423`，通过 `≤+0.005` 标准；冻结、谱系和角色核验通过，protected/test/sealed reads=`0`。独立 E_D macro recall=`0.1355`，低于 `0.50`；每类 recall=`[0.105,0,0.270,0.065,0.090,0.125,0.005,0,0.695,0]`，只有 0/2/5/8 类达到 `0.10`。因此该 TRAIN 邻域样本结果没有通过初步类条件辨识 Gate。

- 训练从 step300 到 step600，共 300 updates；只按 step600 判定。step400/500 不作补救。step600 loss：`L_BOGT=0.363765`、`L_rec=0.051538`、`L_op=0.172780`、`L_sem=2.028325`。S1 anchor/neighbor 到真实 OLS 的标准化距离 mean/q95=`4.2230/9.5332`、`4.2145/9.4200`，仅为诊断量；S1 不证明独立先验新颖性。
- source-bound `preflight_11=READY`，source semantics ID=`f5ec8167b11d1c88c3f92dc8163c66ad1dd396c901150b79eecd271a2ac8529e`；总墙钟=`3808.9 s`（约 63.5 分钟），小于 2 小时预算。
- 报告 SHA-256=`e9814c90bc6600f7cccda1f7ec4e54d560c22d7d67553db37b0e26a7516ab8d4`：[Gate 报告](../../outputs/20260925_m0_bogt_semantic_generation_canary_v1/runs/attempt_01/resolved_report_05.json)、[raw CMOS tensor](../../outputs/20260925_m0_bogt_semantic_generation_canary_v1/runs/attempt_01/generated_raw_cmos.pt)、[复 OLS 输入](../../outputs/20260925_m0_bogt_semantic_generation_canary_v1/runs/attempt_01/generated_s1_ols_coefficients.pt)、[生成接触图](../../outputs/20260925_m0_bogt_semantic_generation_canary_v1/runs/attempt_01/generated_contact_sheet.png)、[固定 DEV 对照接触图](../../outputs/20260925_m0_bogt_semantic_generation_canary_v1/runs/attempt_01/fixed_dev_real_contact_sheet.png)。两图采用统一 pooled 显示尺度，未逐图拉伸对比度。
- 最终科学身份为 `attempt_01`；step600 checkpoint、55,000 行 TRAIN OLS、S1 manifest、复输入、raw tensor、真实 DEV 控制与接触图均已落盘。5 个 post-fit harness recovery 均记录零额外训练更新；所有历史报告保留。`preflight_10` 因三步恢复 trace 不一致被 BLOCKED；源码不变的 `preflight_11` 为 READY。没有 `attempt_02`。
- `CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。不自动延长训练、改采样器、重开 N3、读取 protected/test/sealed 或 promotion。任何后续训练或新采样方案都需新的研究决策与授权。

## 历史状态快照（2026-09-25；N3 actuation correspondence diagnostic 已停止）

### 当时 Delta / 阻塞 / 下一允许动作

Delta：用户已决定停止 N3 structural actuation correspondence diagnostic。该诊断尝试定位参数因子→有效相位→CMOS→阈值支撑/硬结构的局部响应，并比较 `BASE_336`、`FLOOR_ONLY_V1`、`ALIGNED_V1` 三个单步临时臂；目前没有得到完整三臂测量。

状态：原科学身份 `attempt_01` 的首次 episode 因 CUDA/CPU label-index 错误以 `INVALID_HARNESS` 结束。修复后的 source-bound `preflight_recheck_01=READY`；唯一追加的 `attempt_01/recovery_01` 在 `BASE_336` 执行一次临时 Adam 更新并计算更新后测量，随后 `_arm_report` 因 `threshold statistic shape changed` 失败。报告生成发生在 rollback 调用之后；状态记录 rollback 已调用，但 rollback 结果未持久化。不能声称回滚已验证。

- `recovery_01/status.json`：`temporary_updates=1`，`persistent_updates=0`，`checkpoint_writes=0`，`completed_arms=[]`，`automatic_recovery_permitted=false`。`FLOOR_ONLY_V1`、`ALIGNED_V1` 未执行；无 arm report、最终报告或 actuation-chain 分类。`attempt_02` 未创建。
- 根因定位：执行器把 `[N]` 的逐样本 `intensity_changed_pixel_count` 与逐阈值张量一起交给 `_summary_by_class_per_threshold`，该函数要求第二维为三个 threshold。当前这是 harness 报告错误，不构成科学结果。
- repaired-source READY 的 execution source semantics ID=`f7e1e2617db27ab113c3b3d0f1560839a7d732bc4ecd93ee83c6727b6482c6c1`。读取仅涉及诊断固定 TRAIN/N3_FIT；N3_SELECT、N3_CONFIRM、protected/test/sealed reads 均为 0。
- 用户明确停止；当前 `CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。此前单次 recovery 已消费，不能自动再修复重跑、创建新 attempt 或扩展旧授权。未来是否重新测量需先由用户重新决定并授权。
- 证据： [原 attempt 状态](../../outputs/20260924_n3_structural_actuation_target_correspondence_diagnostic_v1/runs/attempt_01/status.json)、[recovery 状态](../../outputs/20260924_n3_structural_actuation_target_correspondence_diagnostic_v1/runs/attempt_01/recovery_01/status.json)、[READY 预检](../../outputs/20260924_n3_structural_actuation_target_correspondence_diagnostic_v1/runs/preflight_recheck_01/preflight.json)。执行器与聚焦测试为 [consumer](../../scripts/research/execute_lightgen_2d_n3_structural_actuation_target_correspondence_diagnostic_v1.py) 和 [tests](../../tests/test_n3_structural_actuation_target_correspondence_diagnostic.py)。

项目主线仍为 `M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION`；本诊断未执行持续训练，未改变 N3 `VALID_FAILED` 结论，也未产生新的语义生成结果。

## 历史交接（2026-09-23；N3 surrogate-to-hard structural 只读归因完成）

## 当前活动 Delta / 阻塞 / 下一允许动作

Delta：N3 rank-2 surrogate-to-hard transfer 的固定 TRAIN 只读归因已完成，状态=`VALID_COMPLETED_DIAGNOSTIC`。报告 classification=`MEASURED_PATTERN_ONLY_NOT_CAUSAL_ATTRIBUTION`；代理 `L_struct` raw improvement=`+0.1267100`，paired 95% CI=`[+0.0993310,+0.1568154]`；337D hard `logρ` raw improvement=`+0.0372387`，CI=`[-0.0361325,+0.1078720]` 跨零。因此只支持该 TRAIN 子集上的代理改善，硬指标总体变化未定，综合机制结论=`MECHANISM_UNRESOLVED`。此结果不改变原 N3 `VALID_FAILED`、所选 rank 或冻结确认结论。

证据身份：contract/protocol SHA=`a5c7c3ab…c53ae16` / `8ffd43fa…5ae8da5`；accepted step9000 SHA=`14e76f32…0afcee`；rank-2 endpoint SHA=`fdeded65…bfeaa7`；诊断样本 128/class、1280 行，index SHA=`fff7aa89…b3c5f2f`。样本只与 N3_FIT/SELECT/CONFIRM 和历史排除行互斥，不宣称独立于 337D reference library。

运行：attempt_01 是前向开始前 heartbeat phase 错误导致的 `INVALID_HARNESS`，不可覆盖；其修复后 `preflight_02=READY`、`attempt_02=COMPLETED`。Rank0/rank2 各 forward 1280 行；optimizer/update/checkpoint=`0/0/0`；DEV、N3角色 forward、protected/test/sealed read=`0`。总耗时=`51.23 s`，峰值 allocated/reserved=`3.40/3.67 GB`。终态 heartbeat 已修复为 `COMPLETED`，历史的 `post_report_status_error.json` 保留。

- [归因报告](../../outputs/20260923_n3_surrogate_hard_transfer_attribution_v1/runs/attempt_02/report.json)，SHA-256=`2558f50f…940e78e`；[执行器](../../scripts/research/diagnose_lightgen_2d_n3_surrogate_hard_transfer_attribution_v1.py) 与 [focused tests](../../tests/test_n3_surrogate_hard_transfer_attribution.py)。
- `CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。本诊断不授权训练、rank 重新选择、N3_CONFIRM 重读或任何后续干预；如需推进，先单独定义并授权新实验。

## 历史 current block（2026-09-22；Harness/Gate 生命周期收敛）

以下内容是 2026-09-22 当时记录，现由上方已完成诊断状态替代：

当时主线和新执行器采用 `PreflightReceipt`、一个 Outer Gate/一个授权，正式 attempt
只能经 `open_scientific_attempt` 进入 `SCIENTIFIC_ACTIVE`；scalar/batched 的 `+10%` 只决定
实现选择，绝对时间/显存与数值等价仍是硬门。

当时阻塞：`attempt_01` 尚未创建，旧 step9046 recovery 不消费；diagnostic-only CUDA 已确认 scalar/batched 均超出绝对资源包络，返回 `PREFLIGHT_BLOCKED(resource)`。

当时下一动作：不再重复同一资源失败的完整 preflight；先完成根因/资源审查。只有资源预算或研究计划发生变化并获授权后，才可重新规划；若 READY，才允许正式 attempt。

## 历史 current block（2026-09-20；Three-evaluator robust semantic continuation）

按用户明确授权 `AUTHORIZED_EXECUTE_THREE_EVALUATOR_ROBUST_SEMANTIC_CONTINUATION_V1`，消费 accepted step3000 Generator、SIC attribution、冻结 E_D 独立确认 evaluator 与冻结 S1 sampler，完成唯一 Outer Gate=`THREE_EVALUATOR_ROBUST_SEMANTIC_CONTINUATION_V1_RESOLVED`。结果=`VALID_COMPLETED`，`outer_gate_passed=true`，classification=`THREE_EVALUATOR_ROBUST_SEMANTIC_CONTINUATION_SUPPORTED`，mechanism=`SEMANTIC_REALIZATION_GAIN_TRANSFERS_TO_REPAIRED_SAMPLED_MANIFOLD`。

`PROJECT MAINLINE=M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION`；`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。

- [report](../../outputs/20260920_three_evaluator_robust_semantic_continuation_v1/runs/attempt_02/report.json)，SHA-256=`1d195aad5307108509349c451aa48f7015b8e27a00bfb66c8e42a56d4c4a0f92`；[step6000 checkpoint](../../outputs/20260920_three_evaluator_robust_semantic_continuation_v1/runs/attempt_02/scientific_step_006000.pt)，SHA-256=`3387e104f7d33e19051a23444b0dabf1eb387c5faca3ce7150bbd1a2212c9986`。
- Stage 1 step3000→3300 完成 `300` updates，E_D delta=`+0.0207770`、paired 95% CI=`[+0.0145900,+0.0270100]`，全部预注册条件通过；Stage 2 随后条件执行至 step6000，额外 `2700` updates。step6000 DEV E_S/E_I/E_C/E_D macro recall=`0.1908/0.5098/0.7332/0.5678`，reconstruction delta=`-0.0036397`，光学有效性通过。
- 冻结 S1 balanced bank 的长程描述性 E_D requested-class macro recall=`0.6250`，`R_raw=1.6869`、`R_feature=0.2968`；latent S1 support=`0.962`（CI lower=`0.953975`）。E_D 仅作 forward-only confirmation/descriptive evaluator，未进入 backward、optimizer 或 checkpoint selection。
- 仅 `generator.phase` 更新；其余 optical/runtime/evaluator/sampler 状态冻结，Adam state 从 step3000 精确恢复；TRAIN/DEV 角色保持不变，protected/test/sealed=`0/0/0`。attempt-01 是 `INVALID_HARNESS` provenance，attempt-02 是唯一有效科学结果。
- 结论边界：支持“语义实现增益转移到已修复的 sampled manifold”这一本次分类；不宣称最终生成质量、硬件 readiness、架构 acceptance 或 promotion。`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`；promotion 前需要新的确认 authority。
- [executor](../../scripts/research/execute_lightgen_2d_three_evaluator_robust_semantic_continuation_v1.py) 与 focused test 已通过 compile、focused pytest（`4 passed`）；未提交 commit，既有 dirty worktree 保留。

## 历史 current block（2026-09-20；SIC independent confirmation evaluator）

按用户明确授权 `AUTHORIZED_ESTABLISH_SIC_INDEPENDENT_CONFIRMATION_EVALUATOR_V1`，消费已接受的
`BLOCKED_NEW_INDEPENDENT_CONFIRMATION_AUTHORITY_UNRESOLVED` 定义结果，执行唯一 Outer Gate=`SIC_INDEPENDENT_CONFIRMATION_EVALUATOR_ESTABLISHED`。
结果=`VALID_COMPLETED`，`outer_gate_passed=true`，classification=`SIC_INDEPENDENT_CONFIRMATION_EVALUATOR_ESTABLISHED`。

`PROJECT MAINLINE=M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION`；`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`；推荐但未授权的下一动作=`AUTHORIZED_EXECUTE_THREE_EVALUATOR_ROBUST_SEMANTIC_CONTINUATION_V1`。

- [report](../../outputs/20260920_sic_independent_confirmation_evaluator_v1/runs/attempt_01/report.json)，SHA-256=`01a875ba775accc5903b58428c1a1e9468e186276ea21ffadf521dc5d8ce5db8`；[frozen E_D candidate](../../outputs/20260920_sic_independent_confirmation_evaluator_v1/runs/attempt_01/final_epoch_candidate.pt)，SHA-256=`b8ece3e3bacedd5b5bec63073bee02ce7489161030b835740c681a1ebd3f9ff9`。
- `E_D=SIC_INDEPENDENT_CONFIRMATION_EVALUATOR_D_V1`，架构为预注册 `PixelMLPConfirmationEvaluatorV1`；架构独立性审计通过，未读取 Generator/optical/CMOS。canonical TRAIN-only FIT/QUALIFICATION=`49,500/5,500`，split seed=`2026092003`，training seed=`2026092004`。
- preflight=`READY`；20 epochs 完整，qualification 一次完成并通过：accuracy/macro recall=`0.9810909/0.9810909`，minimum class recall=`0.9636364`。E_D 未参与 E_S/E_I/E_C/SI/SIC attribution、sampler selection 或 optical hyperparameter selection。
- data firewall：generated CMOS/optical outputs/P2Q2/S1/official test/protected/sealed=`0/0/0/0/0/0/0`；step3000 continuation authority 未消费。下一步必须由新的用户授权启动已定义的 SIC continuation；不得在本任务内自动训练。

## 历史 current block（2026-09-19；Cross-evaluator semantic divergence attribution）

按用户明确授权 `AUTHORIZED_ATTRIBUTE_CROSS_EVALUATOR_SEMANTIC_DIVERGENCE_V1`，从 accepted clean 2L step-0 与 Smooth-guided step-300 完成唯一 Outer Gate=`CROSS_EVALUATOR_SEMANTIC_DIVERGENCE_ATTRIBUTED` 的 TRAIN-only、no-update 归因。结果=`VALID_COMPLETED`，`outer_gate_passed=true`，classification=`SHARED_CROSS_EVALUATOR_LOCAL_SEMANTIC_DIRECTION_SUPPORTED`。

`PROJECT MAINLINE=M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION`；`INPUT_AUTHORITY=CROSS_EVALUATOR_SEMANTIC_DIVERGENCE_ATTRIBUTED`；`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。

- [report](../../outputs/20260919_cross_evaluator_semantic_divergence_v1/runs/attempt_03/report.json)，SHA-256=`c4c390df298ba82563505bec5988b0383709b57e342dfdd623efef9e70425951`；[gradient vectors](../../outputs/20260919_cross_evaluator_semantic_divergence_v1/runs/attempt_03/gradient_vectors.pt)，SHA-256=`75dbe22a1af37e9caa7d6c6e414ff13cf9f66eaa1bedd97953f42b3d5a0107d2`；[directional probes](../../outputs/20260919_cross_evaluator_semantic_divergence_v1/runs/attempt_03/directional_probes.json)，SHA-256=`89597dddfa54a0a58c62c5b5cdca2a73daf8f9f04f6fe67a79d93f5ef4e914bf`。
- Smooth positive control strong (`0.988`)；独立 FrozenMNISTEvaluator canonical validation accuracy=`0.98939997`；两者本轮均为 `DEVELOPMENT_DIAGNOSTIC`，没有 evaluator 进入 training teacher 角色。
- A/B TRAIN bank 各 `20` 行、class-balanced/disjoint；A 构造 output/parameter gradients，B 做 `1e-4/3e-4/1e-3` 对称 phase probes；TRAIN=`810`，DEV/protected/test/sealed=`0/0/0/0`。
- `cos(h_S,h_I)=0.1448393→0.1523234`，`cos(g_S,g_I)=0.3132313→0.0954316`（step-0→step-300）。两个 checkpoint 的 equal-weight normalized joint bisector 都在 B 上同时改善 `L_S` 与 `L_I`，joint optical validity 与 restoration 通过；step-300 单 evaluator directions 有交叉效应。
- no-update：optimizer 未创建、updates=`0`、每次 probe 后恢复且最终恢复 step-0。结论仅为当前冻结 runtime 的局部 shared direction 证据，不是全局训练/硬件/promotion 结论。推荐 successor=`DEFINE_EVALUATOR_ROBUST_SEMANTIC_TRAINING_V1`，不自动执行。
- attempt-01 为 `INVALID_HARNESS` 工程 provenance；attempt-03 为当前消费结果。executor 与 [focused test](../../tests/v2/test_lightgen_2d_attribute_cross_evaluator_semantic_divergence_v1.py) 已通过 compile、Ruff、pytest（`5 passed`）。

## 历史 current block（2026-09-19；Canonical Smooth alignment with independent confirmation）

按用户明确授权 `AUTHORIZED_TRAIN_CANONICAL_SEMANTIC_ALIGNMENT_WITH_INDEPENDENT_CONFIRMATION_V1`，从 accepted clean conflict-aware 2L Generator step-0 完成唯一 Outer Gate=`CANONICAL_SEMANTIC_ALIGNMENT_WITH_INDEPENDENT_CONFIRMATION_V1_RESOLVED` 的 `300`-update Generator-only arm。结果=`VALID_COMPLETED`，`outer_gate_passed=false`，classification=`SMOOTH_GUIDED_ALIGNMENT_WITHOUT_INDEPENDENT_CONFIRMATION`（Case B）。

`PROJECT MAINLINE=M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION`；`INPUT_AUTHORITY=CANONICAL_SEMANTIC_REALIZATION_FAILURE_ATTRIBUTED`；`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。

- [report](../../outputs/20260919_canonical_semantic_alignment_with_independent_confirmation_v1/runs/attempt_01/report.json)，SHA-256=`440230967def4f758a9d72a7999aac34d6127bce487fb078a2320a9b02ab4423`；[preflight](../../outputs/20260919_canonical_semantic_alignment_with_independent_confirmation_v1/runs/attempt_01/preflight.json)，SHA-256=`d38ec15f75dc1d4cf1a9fc9433c934312068843f2be4a6d39970443d3d307d7c`；[step-300](../../outputs/20260919_canonical_semantic_alignment_with_independent_confirmation_v1/runs/attempt_01/scientific_step_000300.pt)，SHA-256=`5efdcfeef9c10a403223c43ae76c464591279261fc593e8e3c1c42016efc1843`。
- 唯一独立 confirmation evaluator=`FrozenMNISTEvaluator v1`，checkpoint SHA=`16a42fad849ed9af249fa65bed9567a8fabc011fcc9d107061f69cdb27f21684`，canonical validation accuracy=`0.98939997`；current teacher 因参与既有 teacher objective 排除，Smooth 因本轮训练 oracle 排除，Shape evaluator 因 `0.9234` qualification failure 排除。confirmation 只测 step `0/300`，未进入 backward/probe。
- Smooth canonical loss=`1.0439515→0.9424558`，Smooth DEV macro recall=`0.1030→0.1088`，paired feature distance delta=`-0.0513843`，centroid matched-nearest delta=`0`；independent confirmation=`0.1698→0.1456`，delta=`-0.0242`，paired bootstrap 95% CI=`[-0.0322,-0.01599]`，因此不支持 independent transfer。
- reconstruction delta=`-0.0016350`（`+0.005` constraint pass）；raw validity 通过；P2+Q2 secondary final `R_raw=2.5831490`、`R_feature=8.2150978`。TRAIN=`3000`、DEV=`20000`、preflight TRAIN=`30`，protected/test/sealed=`0/0/0`；CUDA preflight=`READY`；仅 `generator.phase` 更新。
- 结论边界：只支持 Smooth development objective 的局部改善，不支持 canonical semantic recovery、generation success、architecture/teacher optimality、physical readiness 或 promotion。保持 `CURRENT_AUTHORIZED_NEXT_ACTION=NONE`，不自动追加训练、调参、evaluator search、架构变更或 promotion。
- [executor](../../scripts/research/execute_lightgen_2d_train_canonical_semantic_alignment_with_independent_confirmation_v1.py) 与 [focused test](../../tests/v2/test_lightgen_2d_train_canonical_semantic_alignment_with_independent_confirmation_v1.py) 已通过 compile、Ruff、pytest（`5 passed`）。

## 历史 current block（2026-09-19；Canonical semantic realization failure attribution）

按用户明确授权 `AUTHORIZED_ATTRIBUTE_CANONICAL_SEMANTIC_REALIZATION_FAILURE_V1`，消费 accepted target-image semantic-alignment result、clean step-0 与 failed step-300 Generator checkpoint、FrozenMNISTEvaluator、SmoothMNISTEvaluator 及 TRAIN OLS/sample/target identity，完成唯一 Outer Gate=`CANONICAL_SEMANTIC_REALIZATION_FAILURE_ATTRIBUTED`。结果=`VALID_COMPLETED`，classification=`TEACHER_TO_CANONICAL_OBJECTIVE_TRANSFER_MISMATCH_SUPPORTED`。

`PROJECT MAINLINE=M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION`；`INPUT_AUTHORITY=CANONICAL_SEMANTIC_REALIZATION_FAILURE_ATTRIBUTED`；`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。

- [report](../../outputs/20260919_canonical_semantic_realization_failure_attribution_v1/runs/attempt_03/report.json)，SHA-256=`0efe156ac0bbdf6e7a7d1cfe40f6fd98483fa855f4e7746570f4511fef7668e2`；[gradient vectors](../../outputs/20260919_canonical_semantic_realization_failure_attribution_v1/runs/attempt_03/gradient_vectors.pt)，SHA-256=`c3f1e0f0f4e0abed71946cbc025c254387e9987bc8a6f4301aea4e3c39bf7c43`；[directional probes](../../outputs/20260919_canonical_semantic_realization_failure_attribution_v1/runs/attempt_03/directional_probes.json)，SHA-256=`6271db595c55bebeaebb5f411824b866318e393bd6851a73823d08647068738d`。
- TRAIN-only deterministic class-balanced bank：A/B 各 `20` 行、disjoint、`BANK_SEED=2026091901`；没有 DEV read。`cos(h_T,h_S)=+0.07631/+0.07338`（step-0/300），`cos(g_T,g_S)=+0.15877/+0.10191`，`cos(g_total,g_S)=+0.29854/+0.30322`。
- B symmetric probes show `d_T` transfers to teacher `L_T` but not canonical Smooth `L_S`；`d_S` improves Smooth `L_S` at both step-0 and step-300. Therefore current evidence supports teacher-to-canonical objective transfer mismatch, not absence of a transferable clean-2L canonical local direction.
- `SMOOTH_ROLE_AFTER_ATTRIBUTION=DEVELOPMENT_DIAGNOSTIC`。此前 Smooth measurement remains valid；推荐 successor=`DEFINE_CANONICAL_ALIGNMENT_TRAINING_WITH_INDEPENDENT_CONFIRMATION_V1`，不自动训练，且 successor 必须预注册 distinct independent confirmation evaluator。
- no-update：optimizer 未创建、updates=`0`、probe 参数未持久化、每次 probe 后恢复且最终恢复 step-0；preflight=`READY`。data reads：TRAIN=`800`、DEV/protected/test/sealed=`0/0/0/0`。
- attempt-01/02 为 execution harness stops（无 scientific report/classification），attempt-03 为唯一有效结果。后续保持 `CURRENT_AUTHORIZED_NEXT_ACTION=NONE`，不得自动 teacher search、调参、续训、改架构、替换 evaluator 或 promotion。
- [executor](../../scripts/research/execute_lightgen_2d_attribute_canonical_semantic_realization_failure_v1.py) 与 [focused test](../../tests/v2/test_lightgen_2d_attribute_canonical_semantic_realization_failure_v1.py) 已通过 compile、Ruff、pytest（`3 passed`）。

## 历史 current block（2026-09-19；Generator target-image semantic alignment）

按用户明确授权 `AUTHORIZED_TRAIN_GENERATOR_TARGET_IMAGE_SEMANTIC_ALIGNMENT_V1`，从 accepted clean conflict-aware 2L Generator step-0 执行唯一 Outer Gate=`GENERATOR_TARGET_IMAGE_SEMANTIC_ALIGNMENT_V1_RESOLVED` 的 300-update Generator-only arm。结果=`VALID_COMPLETED`，`outer_gate_passed=false`，classification=`TARGET_FEATURE_ALIGNMENT_WITHOUT_INDEPENDENT_SEMANTIC_TRANSFER`（Case B）。

`PROJECT MAINLINE=M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION`；`INPUT_AUTHORITY=GENERATOR_SEMANTIC_CONTROLLABILITY_ATTRIBUTION_RESOLVED`；`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。

- [report](../../outputs/20260919_generator_target_image_semantic_alignment_v1/runs/attempt_01/report.json)，SHA-256=`500289c9e17e28500e5b103a5d9bfcd7b92b1ace5bd16fdccd9dea683e570f68`；[preflight](../../outputs/20260919_generator_target_image_semantic_alignment_v1/runs/attempt_01/preflight.json)，SHA-256=`a379e1679263b6be41b8adc3e02ce5a1bb30cf3e7a48427c3293edf40f073bfc`；[step-300](../../outputs/20260919_generator_target_image_semantic_alignment_v1/runs/attempt_01/scientific_step_000300.pt)，SHA-256=`7b51f01e0965295f059e0b7f52de1b7c68cdb5cb9ec438c4bca69d4052adcacf`。
- TRAIN `FrozenMNISTEvaluator v1` 的固定 128D pre-classifier feature loss `L_align=` `1.078848→1.046632`；canonical DEV positive control=`0.9876`。仅 `generator.phase` 更新；Encoder/OLS/W/P2/Q2/evaluators 冻结；`delta_align=0.0245538596` 由一次 TRAIN calibration 固定。
- 独立 Smooth DEV=`0.1030→0.1030`，delta=`0.0000`，paired 95% CI=`[-0.0008,+0.0008]`；Smooth paired feature distance=`1.0121→0.9999`，但 centroid matched-nearest=`0.1→0.1` 且仍全部最近 class 8。reconstruction delta=`-0.0014505`，raw validity 通过；P2+Q2 `R_feature=3.0552→3.2780` 未改善。
- role integrity：TRAIN=`3000`、DEV=`20000`，protected/test/sealed=`0/0/0`；preflight=`READY`，updates=`300`，checkpoints/evaluations=`0/100/200/300`，latest=`COMPLETED`。不自动 teacher search、调参、续训、架构变更或 promotion。
- [executor](../../scripts/research/execute_lightgen_2d_generator_target_image_semantic_alignment_v1.py) 与 [focused test](../../tests/v2/test_lightgen_2d_generator_target_image_semantic_alignment_v1.py) 已通过 Ruff、compile、pytest（`5 passed`）。

## Previous current block（2026-09-19；CMOS→Smooth semantic realization）

`PROJECT MAINLINE=M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION`；`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。
`INPUT_AUTHORITY=GENERATOR_SEMANTIC_CONTROLLABILITY_ATTRIBUTION_RESOLVED`。

按用户明确授权 `AUTHORIZED_RESOLVE_CMOS_TO_SMOOTH_SEMANTIC_REALIZATION_V1`，消费 accepted 的同一 5000-row DEV images/labels、raw CMOS、CMOS probe、Smooth checkpoint 与 preprocessing contract，完成唯一 Outer Gate=`CMOS_TO_SMOOTH_SEMANTIC_REALIZATION_RESOLVED`。结果=`VALID_COMPLETED`，最终 classification=`CMOS_CLASS_CODE_PRESERVED_BUT_IMAGE_SEMANTIC_REALIZATION_FAILED`。

- [final report](../../outputs/20260919_cmos_to_smooth_semantic_realization_v1/runs/attempt_04/report.json)，SHA-256=`46fcfd60e5d240a92a9c4ca472f8bb88a90c641474ba11eea9db9da7236dba65`；[finalization manifest](../../outputs/20260919_cmos_to_smooth_semantic_realization_v1/runs/attempt_04/source_manifest.json)，SHA-256=`76299a681ed9a710ec4431152ee1f142553f05b0126de424ee40601fb72871e8`。attempt-04 只修正 attempt-02 的分类谓词，复用 immutable measurement artifacts，没有新 forward。
- Smooth positive control macro recall/accuracy=`0.9880/0.9880`；同一 frozen contract 在 DEV raw CMOS 上 exact reproduce Smooth macro recall=`0.1030`，class-8 recall=`0.998`。contract audit 未发现 double/missing normalization、range/channel/pooling/class-map/order/checkpoint mismatch。
- accepted CMOS linear probe DEV macro recall=`0.9708`；matched transitions：`probe-correct/Smooth-wrong=4359`、反向=`20`。固定 Smooth final embedding 的 normalized feature distances：paired=`1.0121`、same-class reference=`1.0361`、different-class=`1.0655`；CMOS centroids 的 matched-nearest fraction=`0.1`，全部最近 real class 8。最终机制是 class code 仍可线性解码，但 CMOS 未实现 Smooth 所需的 canonical image-semantic manifold。
- role integrity：DEV=`5000`；仅读取 accepted TRAIN raw CMOS=`55000` 以复现既有 probe TRAIN-only standardization；protected/test/sealed=`0/0/0`；optical forward regeneration/backward/update=`false/false/0`；evaluator/probe retraining=`false/false`。optional nonlinear 200-row artifact 未消费，因无 exact 5000-row binding。
- 推荐但未执行的 single successor=`TRAIN_GENERATOR_TARGET_IMAGE_SEMANTIC_ALIGNMENT_V1`；本轮 `CURRENT_AUTHORIZED_NEXT_ACTION=NONE`，不自动训练、调参、续训或 promotion。attempt-01/03 为 aborted provenance；attempt-02 旧分类 report 已被 attempt-04 supersede。
- [executor](../../scripts/research/execute_lightgen_2d_resolve_cmos_to_smooth_semantic_realization_v1.py) 与 [focused test](../../tests/v2/test_lightgen_2d_resolve_cmos_to_smooth_semantic_realization_v1.py) 已通过 Ruff、compile、pytest（`6 passed`）。

## Previous current block（2026-09-19；OLS→CMOS semantic localization）

按用户明确授权 `AUTHORIZED_RESUME_LOCALIZE_OLS_TO_CMOS_SEMANTIC_INFORMATION_V1`，完成一次且仅一次 frozen semantic-localization consumer。唯一 Outer Gate=`OLS_TO_CMOS_SEMANTIC_INFORMATION_LOCALIZED`，结果=`VALID_COMPLETED`，classification=`SEMANTIC_INFORMATION_LOCALIZATION_MIXED`。该 consumer 只物化 clean TRAIN raw CMOS、拟合固定 TRAIN-only multinomial logistic + L2 probes，并在匹配 DEV 上测量；未执行 optical training、parameter update、optimizer/backward through optics、generation 或 promotion。

- [report](../../outputs/20260919_ols_to_cmos_semantic_information_localization_v1/runs/attempt_01/report.json)，SHA-256=`17b7d19ef2528dd2039f94e5e97f7d5e24b54a4dd70047413e9e9daf200c7d98`；[probe metrics](../../outputs/20260919_ols_to_cmos_semantic_information_localization_v1/runs/attempt_01/probe_metrics.json)，SHA-256=`0c63c0485eded59d40fd3cc6374b0912dad8786b143488f5ed1b705c37056724`。
- [TRAIN raw CMOS authority](../../outputs/20260919_ols_to_cmos_semantic_information_localization_v1/runs/attempt_01/train_raw_cmos.pt)：`[55000,1,200,200]` float32，file SHA-256=`1271ba91c396f2a5d730b72f5ab425eff9127b7f987b371e0a0ab13f9f9494cd`，tensor SHA-256=`63efe4bd678440af0a01718a13d7c4dccbb574c4be9c420c84ff47ceaefe3ac8`；clean 2L authority、TRAIN ordering/labels、zero/constant checks 均通过。
- DEV macro recall：OLS=`0.9226`、CMOS=`0.9708`、Smooth=`0.1030`；OLS negative control=`0.1190`。OLS-correct→CMOS-wrong=`56`，反向=`297`；CMOS probe 没有相对 OLS 的可见语义损失，Smooth 则表现为 class-8-dominant output decision mismatch，但在不引入任意 cutoff 的情况下保留 mixed classification。
- data-role integrity：TRAIN=`55000` materialized/consumed，DEV=`5000` measured，protected/test/sealed=`0/0/0`；Smooth 从已物化 DEV CMOS 重算并复现 accepted clean step-0 reference，未重跑 optical model。partial-task resume、流式大 tensor 验证与 dimension harness 修复均为同一 authorization 下的工程 provenance。
- [executor](../../scripts/research/execute_lightgen_2d_ols_to_cmos_semantic_information_localization_v1.py) 与 [focused test](../../tests/v2/test_lightgen_2d_ols_to_cmos_semantic_information_localization_v1.py) 已通过 Ruff、compile、pytest（`5 passed`）。`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`；不得自动启动 coupling repair、encoder retraining、evaluator replacement、generation 或 promotion。

## Previous current block（2026-09-18；latest bounded execution before OLS→CMOS localization）

按用户明确授权 `AUTHORIZED_EXECUTE_MINIMAL_NEAR_IDENTITY_NONLINEAR_GENERATOR_V1`，从 accepted clean 2L conflict-aware Generator step-0（不消费失败/线性 3L endpoint）完成一次 bounded near-identity nonlinear Generator successor。唯一新增变量是在第一层 Generator endpoint 与第二个 phase plane 之间加入冻结 `I_ref` 的 scalar `eta` 算法相位算子：`rho=|E|^2/(I_ref+1e-12)`、`N_eta(E)=E*exp(i*eta*rho)`；`eta=0` 为 exact identity。没有新增 propagation/FFT/crop/aperture/relay/normalization；phase、encoder、OLS、P2/Q2、evaluators、loss weights、数据顺序、Adam family 与 frozen context 均保持不变。它是 MODEL_ASSUMPTION/algorithmic surrogate，不是 Kerr/Te/fiber/hardware validation。结果=`VALID_COMPLETED`，Outer Gate=`MINIMAL_NEAR_IDENTITY_NONLINEAR_GENERATOR_V1_RESOLVED`，`outer_gate_passed=false`，classification=`NONLINEAR_LOCAL_GAIN_WITHOUT_DATASET_RECOVERY`（Case B）。

- [report](../../outputs/20260918_minimal_near_identity_nonlinear_generator_v1/runs/attempt_02/report.json)，SHA-256=`d2526b49540d6ea03483f51413587b09b8195440af3c72cd1925f3673c7f5673`；[preflight](../../outputs/20260918_minimal_near_identity_nonlinear_generator_v1/runs/attempt_02/preflight.json)，SHA-256=`fc1bf02eb6dd62ffd1b4d61b671172c8af85069fe456a5c78b1ceda232179b08`；[step-300](../../outputs/20260918_minimal_near_identity_nonlinear_generator_v1/runs/attempt_02/scientific_step_000300.pt)，SHA-256=`630447422b1cac7ad900a98b25712ed53642b2c98f4ca64e7bde3617694bd32a`。
- 起点 authority：conflict-aware report SHA=`dc05742fae2333e21a1d123fd93d4032b9e67676ae4e424ea3c68e5d6db8c904`、accepted clean step-0 SHA=`a61dd84728aa220aa567ab6cd5f9daa788b21e76be3c506054c2b7c0a23dcd34`；failed/linear 3L step-300 未加载。`I_ref=0.9120797964019747`，fixed TRAIN calibration bank=`10`，insertion=`lightgen_generator_layer_1` at `z=0.6295 m`。
- eta=0 function-preserving=`accepted_for_stage1=true`，所有报告的 field/raw/evaluator/reconstruction 比较 `max_abs=0`。Stage 1 A/B eta gradients=`-0.4071133/-1.1347258`，方向=`+1`，同号且 beneficial；B 三组 symmetric probe Smooth slopes=`-1.4567375/-1.1968613/-1.1153221`，raw validity 全通过。对应 [gradient artifact](../../outputs/20260918_minimal_near_identity_nonlinear_generator_v1/runs/attempt_02/stage1_gradient_vectors.pt)，SHA-256=`14b83d3ae8b2fc3c16ca283d789d7fbdae7ce393677a6b01b33fbf0f667d001b`。
- `READY` preflight、3-step real update、negative control、frozen firewall、optimizer state boundary 均通过；300 updates，checkpoints/DEV=`0/100/200/300`，CUDA=`true`；reads：TRAIN=`10+30+3000`，DEV=`20000`，protected/test/sealed=`0`。
- Smooth DEV=`0.1030→0.1028→0.1028→0.1024`，delta=`-0.0006`，paired 95% CI=`[-0.0020,+0.0006]`，dataset semantic recovery unsupported；reconstruction=`0.0687409→0.0671431`，delta=`-0.0015977`，约束通过。P2+Q2 transfer：`R_feature=3.0552277→1.5064399`，但 `R_raw=1.5288204→1.8741536`；raw finite/nonconstant=`1.0/1.0`、zero-image=`0`。
- attempt-01 在科学更新前因 logical-batch harness 边界 fail-closed（scientific=`0`）并保留；最小执行器修复后 attempt-02 完成，不构成新的 scientific arm。无 retry/sweep/调参/续训；不支持 generation success、physical hardware validity 或 promotion。`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。
- [executor](../../scripts/research/execute_lightgen_2d_minimal_near_identity_nonlinear_generator_v1.py) 与 [focused test](../../tests/v2/test_lightgen_2d_minimal_near_identity_nonlinear_generator_v1.py) 已通过 compile/lint/pytest（`3 passed`）。

## Minimal Generator capacity expansion（2026-09-18；previous bounded execution）

按用户明确授权 `AUTHORIZED_EXECUTE_MINIMAL_GENERATOR_CAPACITY_EXPANSION_V1`，从 accepted 2L conflict-aware Generator endpoint 完成一次且仅一次 minimal 3L capacity-expansion successor。唯一结构变量是把现有第二个 `1.50 mm` interval 对称拆为 `0.75 mm + 0.75 mm`，插入初始为零的 trainable phase plane；其余 optical/runtime/frozen context、数据顺序与 optimizer family 均冻结，旧 Adam moments 复制、新平面 moments 置零。结果=`VALID_COMPLETED`，Outer Gate=`MINIMAL_GENERATOR_CAPACITY_EXPANSION_V1_RESOLVED`，`outer_gate_passed=false`，classification=`LOCAL_CONTROLLABILITY_GAIN_WITHOUT_DATASET_RECOVERY`（Case B）。

- [report](../../outputs/20260918_minimal_generator_capacity_expansion_v1/runs/attempt_01/report.json)，SHA-256=`ea8fb6b3e766a667461cfa061d9d22f42306d34eb0757aeb0d0e87d6202b817f`；[preflight](../../outputs/20260918_minimal_generator_capacity_expansion_v1/runs/attempt_01/preflight.json)；[step-300 checkpoint](../../outputs/20260918_minimal_generator_capacity_expansion_v1/runs/attempt_01/scientific_step_000300.pt)，SHA-256=`7ee9c3782b3b5ddff1ebf064eef4836d7e18a63385fff0dbf64bbb3b2cbac517`。
- function-preserving check=`accepted_for_stage1=true`；Stage 1 `supported=true`，new-plane Smooth gradient norm=`0.1424059`，3L/2L fixed-scale Smooth response ratios=`1.1105/1.0947/1.1173`；preflight=`READY`，CUDA=`true`，TRAIN/DEV=`3000/20000`，protected/test/sealed=`0`。
- Smooth DEV=`0.1030→0.1036→0.1034→0.1032`，delta=`+0.0002`，paired 95% CI=`[-0.0004,+0.0008]`，semantic gate 未支持；reconstruction=`0.0687409→0.0663280`，delta=`-0.0024129`，约束通过；raw validity=`pass`。
- exact P2+Q2 transfer：`R_raw=1.5288204→2.0526421`、`R_feature=3.0552277→3.6763498`，而 `R_u=1.5095471`、`R_z=1.8283132` 保持；expanded CMOS SHA-256=`ebe6e7b1bae3df94d474fbb4d6e4299e2bd1671e769c2d68697e5d7fdc1c968c`。
- 结论仅为当前 frozen Generator 的局部 controllability gain；不支持 dataset-level semantic recovery、global capacity sufficiency、generation success 或 promotion。`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`；不自动追加新 arm、sweep、调参、续训或架构变更。未完成 preflight 目录保留为 aborted provenance，不作为科学结果。

## Conflict-aware Generator semantic recovery（2026-09-18；previous bounded execution）

按用户明确授权 `AUTHORIZED_EXECUTE_CONFLICT_AWARE_GENERATOR_SEMANTIC_RECOVERY_V1`，从 accepted frozen-latent semantic-recovery successor step-0 启动一次且仅一次 conflict-aware Generator-only successor。唯一科学变量是 gradient composition：`g_total=g_sem+g_rec_safe+g_op_safe`；固定 `eps_proj=1e-12`，仅移除与 semantic gradient 负内积的 reconstruction/optical-power 分量。结果=`VALID_COMPLETED`，Outer Gate=`CONFLICT_AWARE_GENERATOR_SEMANTIC_RECOVERY_V1_RESOLVED`，classification=`CONFLICT_AWARE_SEMANTIC_RECOVERY_NOT_SUPPORTED`（Case C）。

- [report](../../outputs/20260918_conflict_aware_generator_semantic_recovery_v1/runs/attempt_01/report.json)，SHA-256=`dc05742fae2333e21a1d123fd93d4032b9e67676ae4e424ea3c68e5d6db8c904`；[preflight](../../outputs/20260918_conflict_aware_generator_semantic_recovery_v1/runs/attempt_01/preflight.json)；[step-300 checkpoint](../../outputs/20260918_conflict_aware_generator_semantic_recovery_v1/runs/attempt_01/scientific_step_000300.pt)，SHA-256=`9e561f4a95da2a6962e6e6bcf0ac3ebb0680e974a8ef85c2c99cf18425b73d35`；[P2+Q2 CMOS](../../outputs/20260918_conflict_aware_generator_semantic_recovery_v1/runs/attempt_01/successor_p2q2_generated_cmos.pt)，SHA-256=`dfc6eb2e1744314f4518aef3b3c734c60ce80b728293741421c4d923c3fa7f44`。
- `READY` preflight、3-step real update、negative control、frozen firewall、same optimizer state/data order/weights、projection orthogonality 全通过；`beta=2.4509953404`、`gamma=0.5330424285`、`delta=0.0065783691`；old TN/scalar weighted-sum backward=`false`。
- Smooth=`0.1030→0.1034→0.1038→0.1036`，delta=`+0.0006`，paired 95% CI=`[0,+0.00121]`；CI lower 未 `>0`。reconstruction=`0.0687409→0.0673604`，delta=`-0.0013805`；constraint/raw validity=`pass/pass`。
- P2+Q2 `R_raw=1.5288204→1.5574754`、`R_feature=3.0552277→3.1940135`，feature 未下降；class histogram=`[1,3,48,68,0,44,1,18,10,7]`、dominant=`0.34`、normalized entropy=`0.71321`。TRAIN/DEV=`3000/20000`，protected/test/sealed=`0`。
- 结论只支持：在当前 frozen Generator 参数化下，移除局部 auxiliary gradient conflict 不足以恢复 dataset-level semantics；不自动修改 architecture。`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。

## Generator semantic-controllability attribution（2026-09-18；accepted input authority）

按 accepted semantic-recovery result 完成一次且仅一次 no-update attribution consumer。结果=`VALID_COMPLETED`，Outer Gate=`GENERATOR_SEMANTIC_CONTROLLABILITY_ATTRIBUTION_RESOLVED`，primary=`GENERATOR_SEMANTIC_CONTROLLABILITY_MIXED`，secondary=`RECONSTRUCTION_SEMANTIC_GRADIENT_CONFLICT_SUPPORTED`。结论只覆盖 accepted calibration batch 与 frozen runtime；不构成 global architecture、generation acceptance、physical validity 或 promotion 结论。

- [attribution report](../../outputs/20260918_generator_semantic_controllability_attribution_v1/runs/attempt_01/report.json)，SHA-256=`3304262b2cd5fbff22e423c415984c14eeb1a2906871ad45e3a62265479d89b5`；[gradient vectors](../../outputs/20260918_generator_semantic_controllability_attribution_v1/runs/attempt_01/gradient_vectors.pt)，SHA-256=`ae4515640b0005b9e3c56328d022c430e97ecc3cdba0a9dcce5f6b29c7289da1`；[directional probes](../../outputs/20260918_generator_semantic_controllability_attribution_v1/runs/attempt_01/directional_probes.json)，SHA-256=`fce20780ae2c194b3e81880acbeb4a6c2a5d9fec352b8559487da3ccbc9e0c18`。
- step-0 norms=`0.0005866184 / 0.2936746160 / 0.2829194241`（reconstruction / TRAIN semantic / Smooth semantic）；step-300=`0.0005759371 / 0.2730509876 / 0.2779645008`。cosines step-0=`C_train_smooth +0.18470490`、`C_rec_smooth -0.06370815`、`C_rec_train +0.07054067`；step-300=`+0.11107681`、`-0.08296854`、`+0.04521843`。
- `eps={1e-4,3e-4,1e-3}` symmetric probes 全部恢复；`-g_rec` 改善 reconstruction 但恶化 Smooth，`-g_sem_train`/`-g_sem_smooth` 改善 Smooth；raw validity=`finite/nonconstant 1.0/1.0`、zero-image=`0`。
- mismatch/controllability 均未被局部证据单独隔离；机制保持 mixed。推荐 single repair direction=`NONE`，不自动训练、调参、续训或 promotion。no-update=`optimizer false / updates 0 / runtime restored true`；TRAIN calibration reads=`10`，DEV/protected/test/sealed=`0`。
- [executor](../../scripts/research/execute_lightgen_2d_generator_semantic_controllability_attribution_v1.py) 与 [focused test](../../tests/v2/test_lightgen_2d_generator_semantic_controllability_attribution_v1.py) 已通过 compile/lint/pytest。`PROJECT MAINLINE=M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION`；`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。

## Frozen latent Generator semantic-recovery successor（2026-09-18；accepted input authority）

本轮用户明确授权 `AUTHORIZED_EXECUTE_FROZEN_LATENT_GENERATOR_SEMANTIC_RECOVERY_V1`，从接受版 Chen BOGT step-300 状态完成一次且仅一次 frozen-latent semantic-recovery successor。仅 `generator.phase` 可训练；encoder/OLS/scale/`T=I_100`/`W_OLS`/latent normalization/P2/Q2/evaluators 冻结；执行 `300` 个 TRAIN updates，并在 `0/100/200/300` 完成 full DEV 诊断。结果=`VALID_COMPLETED`，Outer Gate=`FROZEN_LATENT_GENERATOR_SEMANTIC_RECOVERY_V1_RESOLVED`，classification=`GENERATOR_SEMANTIC_RECOVERY_NOT_SUPPORTED`。

- [successor report](../../outputs/20260918_frozen_latent_generator_semantic_recovery_v1/runs/attempt_01/report.json)，SHA-256=`fcb85e17e4b3d258fbe76d975103869e8d18a11252965dd5581f02a0acdb19b3`；[preflight](../../outputs/20260918_frozen_latent_generator_semantic_recovery_v1/runs/attempt_01/preflight.json)；[step-300 checkpoint](../../outputs/20260918_frozen_latent_generator_semantic_recovery_v1/runs/attempt_01/scientific_step_000300.pt)。
- preflight=`READY`：3-step real update 通过，full-runtime optimizer negative control 被拒绝；Adam 仅更新 `generator.phase`，并恢复接受版 generator moments。目标为 `L_recovery=beta*L_rec+gamma*L_op+delta*L_sem`，`beta=2.4509953404`、`gamma=0.5330424285`、`delta=0.0065783691`；旧 TN term 未进入 backward。
- 独立 Smooth DEV macro recall=`0.1030→0.1034`，delta=`+0.0004`，paired bootstrap `B=999`、seed=`271828`、95% CI=`[-0.0004,+0.0012]`，语义支持失败；reconstruction=`0.0687409→0.0673416`，delta=`-0.0013992`，约束通过。
- exact P2+Q2/real-control transfer 的 `R_raw=1.5288204→1.5524823`、`R_feature=3.0552277→3.2092072`，R_feature 未下降；raw finite/nonconstant=`1.0/1.0`、zero-image=`0`，光学 validity 通过。TRAIN/DEV reads=`3000/20000`，protected/test/sealed=`0`。
- 未改变 architecture、OLS、P2/Q2、evaluators，也未触及 promotion；[executor](../../scripts/research/execute_lightgen_2d_frozen_latent_generator_semantic_recovery_v1.py) 与 [focused test](../../tests/v2/test_lightgen_2d_frozen_latent_generator_semantic_recovery_v1.py) 已通过 compile/lint/pytest。该结果只覆盖当前 frozen-latent、generator-only、300-update scope，不支持 semantic recovery 或 promotion。
- `PROJECT MAINLINE=M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION`；phase authority=`CHEN_BOGT_PHASE_COMPLETION_REPAIR_V1_RESOLVED` 保持有效；`CURRENT_AUTHORIZED_NEXT_ACTION=NONE`。不自动追加 arm、调参、续训或 promotion。

## Chen BOGT phase-completion repair（2026-09-18；accepted input authority）

本 successor 消费的 phase authority 是已接受的 frozen-model downstream propagation diagnostic：Stage 1 在 CPU 上完成 Q0/Q1/Q2/Q3 与 Q4 control 的 `B=999` within-pair calibration，Stage 2 对选中的 Q2 完成 `200` 个 selected-generated direct Generator forwards，并 exact reuse `200` 个 real-control CMOS；未训练、未 refit `B/C/L`、未改变 P2/OLS/Generator/evaluator。

- [phase repair report](../../outputs/20260918_chen_bogt_phase_completion_repair_v1/runs/attempt_01/phase_completion_repair_report.json)；[phase selection](../../outputs/20260918_chen_bogt_phase_completion_repair_v1/runs/attempt_01/phase_selection.json)；[selected latent bank](../../outputs/20260918_chen_bogt_phase_completion_repair_v1/runs/attempt_01/selected_phase_completion_latent_bank.pt)。
- 选中 Q2=`ANCHOR-CONDITIONED_JOINT_UNCONDITIONAL`；冻结 P2+Q0 → Q2 的 `R_u=1.5095→1.5095`、`R_phi=7.1200→0.9937`、`R_z=19.0598→1.8283`、`R_raw=2.1189→1.5288`、`R_feature=7.4096→3.0552`，classification=`DOWNSTREAM_DIRECTIONALLY_IMPROVED`。
- selected raw validity 为 finite/nonconstant=`1.0/1.0`、zero-image=`0`；该 authority 只覆盖当前 frozen model/frozen bank 的 downstream propagation，不构成 generation acceptance、physical validity 或 promotion。

## Chen latent-prior support decomposition（2026-09-18；historical source authority, superseded by the compatibility calibration above）

attempt_08 的 frozen TRAIN-derived decomposition 事实保持有效：`R_TN=94.4553`、`R_independent=9.3665`、`R_copula=1.7113`、`R_real_row_control=1.2407`，描述性 marginal/dependence reduction=`0.9105`/`0.9150`；其 strict-Pareto primary diagnosis=`UNRESOLVED`、candidate=`NONE` 只表示上一轮 selector 的结果，已被本轮预先定义的 finite-sample primary compatibility rule supersede，不代表 P1/P2 工件被修改。

## Chen BOGT sampled-generation canary（2026-09-18；historical input authority, superseded by the compatibility calibration above）

此前固定 `N_LATENT=1000`、`N_OPTICAL=200` 的 canary 事实保持有效；其 primary diagnosis=`MULTI_STAGE_MISMATCH`，prior/support `R_u=85.1354`，phase `R_phi=7.1200`，full-complex-z `R_z=27.7766`，raw/feature `R_raw=2.8473`/`R_feature=8.2326`。该 canary 是本轮 P0 identity 的冻结输入，不再代表当前 latest bounded execution；其 raw CMOS validity/diversity 与无 labels 因而未计算 accuracy/recall 的事实保持不变。

## Chen BOGT training / generation-ready bundle（2026-09-18；historical, superseded by the current compatibility calibration above）

前一轮固定 `300`-update training、final TRAIN collection、wrapped-phase model fit 与 generation-ready bundle 事实保持有效；其 `VALID_COMPLETED_GENERATION_BUNDLE_READY` 只描述 bundle-ready 状态，已被上方 compatibility calibration 的 current block supersede。step-300 DEV diagnostic semantic macro recall=`0.1474<0.937` 的历史事实不变；本节不得覆盖上方 current block。

## 当前 research fast path / long-run resilience（2026-09-17；governance-only；historical）

本轮 Outer Gate=`RESEARCH_FAST_PATH_AND_LONG_RUN_RESILIENCE_V1_ESTABLISHED`；只把普通 `R0/R1/R2` research 的一个 Outer Gate、一个授权、工程预检、heartbeat、
progress、operational recovery 与同轨迹 resume 规则落到共享 harness，并迁移当前 Chen executor
的工程入口；没有 scientific update/run、generation、CUDA 或 evidence（仅运行工程预检 tiny checks）。项目主线仍是
`M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION`，当前授权 next action 仍为 `NONE`。下方历史 details
保留原事实，不得覆盖本节或本页的 current block。

## 4L Linear optimization-ceiling continuation（2026-09-16，VALID_COMPLETED；C3000 plateau）

已从已验证的 C=`4L Linear` local step=`2000` 唯一续训 `1000` 个 CUDA updates 到 step=`3000`；仅 `generator.phase` 可训练，4 层 generator、3 mm 层间距、2048 grid、linear facet、rank-9 adapter、encoder、complex 100-port OLS、`W_OLS`、objective、数据角色和 C2000 Adam moments 均保持不变。每 200 步 checkpoint/block/progress 均已保留；C3000 DEV=`5000`、cyclic=`1000` 完整评估，未进入 C4000。preflight attempt-01/02 为训练前 `invalid_harness` 历史，attempt-03=`AUTHORIZABLE`。

- [contract](../../outputs/20260916_4l_linear_optimization_ceiling_v1/runs/4l_linear_optimization_ceiling_v1_contract_attempt_03/4l_linear_optimization_ceiling_v1_contract.v1.json)，SHA-256=`5a18f32c7e8f66566b6554f0a6adc189a93087528a40820c266c9793a68e201e`；[preflight](../../outputs/20260916_4l_linear_optimization_ceiling_v1/runs/4l_linear_optimization_ceiling_v1_preflight_attempt_03/4l_linear_optimization_ceiling_v1_preflight_report.v1.json)，SHA-256=`09a797e8155ef01dde45be656c65a7a0002f95b652260075ee854f49cbdf643c`。
- [aggregate](../../outputs/20260916_4l_linear_optimization_ceiling_v1/runs/4l_linear_optimization_ceiling_v1_aggregate_attempt_01/4l_linear_optimization_ceiling_v1_aggregate_report.v1.json)，SHA-256=`8687ac41431917485806ea3b3f188673544e6e6945808915db10f07212b96793`；lineage SHA-256=`a6eca15e88ff59851379498389a52a67ea03c2c890b13420d5c5bb3fe70ac6bc`。
- [C3000 checkpoint](../../outputs/20260916_4l_linear_optimization_ceiling_v1/runs/4l_linear_optimization_ceiling_v1_c_linear_continuation_attempt_01/c_4l_linear_continuation_step_03000_checkpoint.v1.pt)，SHA-256=`b5369c6ad16391adf772ba8aa42ea9642048a52041b07d50ad0e3c4928cc6270c`。

结果：`R2000=0.5774 → R3000=0.5790`，`Delta_R=+0.0016`，paired bootstrap 95% CI=`[-0.0002,+0.0034]`；`L_image=0.0467247652→0.0469439335`（delta=`+0.0002191683`）；raw validity/cyclic transfer=`pass/pass`。由于 C3000 未达到 `Delta_R>=+0.010` 且 CI lower `>0`，预注册 decision=`continue_to_4000=false`、classification=`4L_LINEAR_CONTINUATION_PLATEAU_SUPPORTED`。各类 recall 变化为 `0:+0.000`、`1:+0.000`、`2:+0.002`、`3:+0.002`、`4:+0.004`、`5:+0.008`、`6:+0.010`、`7:-0.008`、`8:+0.000`、`9:-0.002`；1 类仍为零。

实际续训耗时约 `924.07 s`，RTX 3060 Laptop、grid=`2048`；未执行 ReLU、Stage B、generation、physical Te validation、protected/test/sealed 或 formal promotion。结论只覆盖当前 generator-only 优化路径，不声称全局优化最优或真实 Te 薄膜可行。实验完成后已关闭半小时 heartbeat 监控；后续材料参数实验需新的 contract/授权。

## Strict TRAIN-calibrated ReLU90 successor（2026-09-15，VALID_COMPLETED；Outer Gate NOT_SUPPORTED）

按用户明确授权 `EXECUTE_STRICT_TRAIN_CALIBRATED_RELU90_GAIN_V2` 完成一次有效 successor：attempt-03 preflight=`AUTHORIZABLE`，A 复用上一轮 nominal `A_LINEAR` 的完整 exact-reuse authority，B 使用 disjoint TRAIN-only 标定的全局 ReLU90 facet 完成 `2000` updates。attempt-01/02 仅为训练前 harness INVALID 历史，未发生 scientific update。

- [contract](../../outputs/20260915_strict_train_calibrated_relu90_gain_v2/runs/strict_train_calibrated_relu90_gain_v2_contract_attempt_03/strict_train_calibrated_relu90_gain_v2_contract.v2.json)，SHA-256=`b4488885997f65ad21a2c34c97613f25d3dd9a82f3d95c63ec0838d5537bfb1b`；[preflight](../../outputs/20260915_strict_train_calibrated_relu90_gain_v2/runs/strict_train_calibrated_relu90_gain_v2_preflight_attempt_03/strict_train_calibrated_relu90_gain_v2_preflight_report.v2.json)，SHA-256=`14ae74b7f7a0dedc7da8261b4bb475d758b46fc3202a78fa4d883d0c00dc71de`。
- TRAIN calibration=`1000`、disjoint TRAIN validation=`1000`，`tau=0.10682560942718147`，`eta_cal=0.9000000000`，`eta_train_valid=0.9015109995`；无 DEV threshold selection、无 OLS full cache。
- A reuse authority：`R2000=0.6170`、`L_image=0.04578155`。B：`R0=0.5454`、`R2000=0.6138`、`L_image 0.04967835→0.04630363`。
- [aggregate](../../outputs/20260915_strict_train_calibrated_relu90_gain_v2/runs/strict_train_calibrated_relu90_gain_v2_aggregate_attempt_03/strict_train_calibrated_relu90_gain_v2_aggregate_report.v2.json)，SHA-256=`def0a60c7176b82bbb776b4c6989da8185b744c5d4504b1eec1995b9d4344ca7`；classification=`STRICT_TRAIN_CALIBRATED_RELU90_NOT_SUPPORTED`；`Delta_R=-0.0032`，paired 95% CI=`[-0.00641,0]`，`Delta_L_image=+0.00052208`，cyclic/raw=`pass/pass`。

当前解释边界：ideal algorithmic ReLU surrogate 的严格 TRAIN 标定没有支持相对 nominal A 的语义增益；facet activation 不是实测 Te 曲线。Stage-A 最高 `R=0.6170<0.937`，Stage B、generation、physical Te validation、protected/test/sealed 与 promotion 均继续关闭。当前没有待执行的自动 follow-up，因此 `CURRENT_AUTHORIZED_NEXT_ACTION=NONE`；如需继续，必须另行定义并授权新的 Gate。

## N6 nonlinear fiber-facet activation canary（2026-09-14，VALID_COMPLETED；Outer Gate NOT_SUPPORTED；one-shot A/B）

N5-V2 已作为线性 optical-depth 主线终止：`OPTIMIZATION_HORIZON_DOMINANT_TRUE_DEPTH_NOT_SUPPORTED`，不再追加线性深度候选。按用户明确授权 `EXECUTE_N6_NONLINEAR_FIBER_FACET_ACTIVATION_CANARY_V1`，在
`research/lightgen-ols-native-semantic-generation-v1`、HEAD=`a9cd1749adfa690e282fd52a6425564fc2cf102c` 上，复用 exact N5-A step-300 endpoint（global step=`600`），完成一次且仅一次 A/B CUDA scientific scope，各 `300` updates，global step=`600→900`。A 是 linear continuation control；B 是放在 canonical `lightgen_generator_entrance` facet 的固定 algorithmic nonlinear surrogate，不是实测材料曲线，不能据此宣称 physical material validation。两臂 rank=`9`，encoder/OLS/`W_OLS` 冻结，仅 adapter 与现有 generator phase 更新。

- authority=`DEFINE_N6_NONLINEAR_FIBER_FACET_ACTIVATION_CANARY_V1`；[contract](../../outputs/20260914_n6/runs/lightgen_2d_nonlinear_fiber_facet_activation_canary_contract_v1_attempt_02/lightgen_2d_nonlinear_fiber_facet_activation_canary_contract.v1.json)，SHA-256=`1943c9327076e57451923ef3d0bc6815b9f336a5e7fc8efc2b6348702192bfd5`。
- preflight attempt-01/02/03=`INVALID_HARNESS`（scientific=`0`，保留为工程历史）；attempt-04=`AUTHORIZABLE`；[report](../../outputs/20260914_n6/runs/lightgen_2d_nonlinear_fiber_facet_activation_canary_preflight_v1_attempt_04/lightgen_2d_nonlinear_fiber_facet_activation_canary_preflight_report.v1.json) SHA-256=`b16742d3b3805293ff2cc8a06d92b32c494cab0e07837fa22056cac55f634aa6`；lineage SHA-256=`7d1c63c360a45a813575aeb3b5a3d47d265073f85f86e10154007a6ef983f07c`。
- A [report](../../outputs/20260914_n6/runs/lightgen_2d_nonlinear_fiber_facet_activation_canary_a_attempt_01/lightgen_2d_nonlinear_fiber_facet_activation_canary_a_linear_continuation_control_scientific_report.v1.json) SHA-256=`7ec8e7c0d02f46cef6ac36dbaa24cf1a8a70a8ebbbce05a7057a76501510bc8c`；B [report](../../outputs/20260914_n6/runs/lightgen_2d_nonlinear_fiber_facet_activation_canary_b_attempt_01/lightgen_2d_nonlinear_fiber_facet_activation_canary_b_nonlinear_facet_scientific_report.v1.json) SHA-256=`8763b1547f8f9a4db102f0d987bca8ac1c76048cb688aaf6031e52b1784233c7`。
- [aggregate report](../../outputs/20260914_n6/runs/lightgen_2d_nonlinear_fiber_facet_activation_canary_aggregate_v1_attempt_01/lightgen_2d_nonlinear_fiber_facet_activation_canary_aggregate_report.v1.json) SHA-256=`56534572368253dd0c488a695e31050a240a4b3e09535c8241eeff4c4a676ce7`；[lineage](../../outputs/20260914_n6/runs/lightgen_2d_nonlinear_fiber_facet_activation_canary_aggregate_v1_attempt_01/lightgen_2d_nonlinear_fiber_facet_activation_canary_aggregate_lineage.v1.json) SHA-256=`ad31df72f6edc00d8d6ad8e8d6166f7160c614ab5d270f0aee1d9a7cfa7a560b`；classification=`N6_NONLINEAR_FIBER_FACET_ACTIVATION_NOT_SUPPORTED`，aggregate `VALID_COMPLETED`。

关键结果：

| candidate | Smooth 0/100/200/300 | DEV `L_image` 0→300 | `R300` | cyclic/raw |
| --- | --- | ---: | ---: | --- |
| A linear control | 0.5526 / 0.5518 / 0.5630 / 0.5788 | 0.05241717→0.04968191（Δ=-0.00273526） | 0.5788 | pass / pass |
| B nonlinear facet | 0.5514 / 0.5530 / 0.5632 / 0.5800 | 0.05093504→0.04988193（Δ=-0.00105311） | 0.5800 | pass / pass |

`Delta_NL=+0.0012`，paired bootstrap 95% CI=`[-0.0014,+0.0040]`，未达到 `+0.05` 与 CI lower `>0`，Outer Gate 不支持。B facet step-300 activation=`0.66219`、suppressed=`0.33781`、mean power ratio=`0.97556`；B reconstruction 约束通过。Stage-A firewall 独立失败：A=`0.5788<0.937`、B=`0.5800<0.937`；Stage B、generation、promotion 继续关闭。无 retry、无新 optical depth、无 rank/loss/lambda/lr 调整、无 protected/test/sealed read。

## N5 V2 true-optical-depth comparison（2026-09-14，VALID_COMPLETED；one-shot A/B）

用户明确授权 `EXECUTE_N5_SUCCESSOR_TRUE_OPTICAL_DEPTH_COMPARISON_V2`。在
`research/lightgen-ols-native-semantic-generation-v1`、HEAD=`a9cd1749adfa690e282fd52a6425564fc2cf102c`
上完成一次且仅一次 V2 bounded preflight、A/B 各 `300` 个 CUDA updates 与冻结 paired
aggregate。V2 是 V1 `INVALID_N5_SCIENTIFIC_UNEXPECTED_EXECUTION_ERROR` 的独立 successor，
不是 retry/continuation；V1 scientific metrics 与未持久化 observations 均未消费。A 为
continued C1（rank-9 OLS interface + 当前 2-layer generator）；B 在相同起点上加入
full-`2048` grid、`0.75 mm` midpoint phase、`0.75 mm` propagation factorization。encoder、
OLS、`W_OLS`、Smooth evaluator、数据角色、objective、optimizer、顺序与预算均冻结。

- authority=`DEFINE_N5_SUCCESSOR_TRUE_OPTICAL_DEPTH_COMPARISON_V2_CONTRACT`；[contract](../../outputs/20260914_n5_v2/runs/lightgen_2d_true_optical_depth_comparison_v2_contract_v1_attempt_01/lightgen_2d_true_optical_depth_comparison_v2_contract.v1.json)，SHA-256=`f7928d640093d8133380233c0ee451221b93f1ee03236b381227228a5c1df9fb`。
- preflight attempt-03 [report](../../outputs/20260914_n5_v2/runs/lightgen_2d_true_optical_depth_comparison_v2_preflight_v1_attempt_03/lightgen_2d_true_optical_depth_comparison_v2_preflight_report.v1.json)，SHA-256=`86933442d9051d6a11210b08c14cb59b9c9e1f2f229d2feaa75f0ed2a2d11992`；lineage SHA-256=`8a0f169fa41edceca19ffd271118b55cc5455badd6e24772aa4849665218d519`；`AUTHORIZABLE`，scientific=`0`。
- A [report](../../outputs/20260914_n5_v2/runs/lightgen_2d_true_optical_depth_comparison_v2_candidate_a_attempt_01/lightgen_2d_true_optical_depth_comparison_v2_a_continued_c1_scientific_report.v1.json)，SHA-256=`ed796d14971aa8cf44f708ba3dc274601f38db46b82559dc4572bcd202d14fca`；lineage SHA-256=`a77fb0b58afc6f8d0a53c45cc68f362051f1a153abc119b413183cdab70361f8`；`VALID_COMPLETED`。
- B [report](../../outputs/20260914_n5_v2/runs/lightgen_2d_true_optical_depth_comparison_v2_candidate_b_attempt_01/lightgen_2d_true_optical_depth_comparison_v2_b_true_depth_c1_scientific_report.v1.json)，SHA-256=`8f3bc38c927a64f97431325504860bd71c622cd989e248022fe736706c685681`；lineage SHA-256=`d092f47bef7021c9bae73423a4a8af19b54b1449ac448beebca691265dfe91e5`；`VALID_COMPLETED`。
- [aggregate report](../../outputs/20260914_n5_v2/runs/lightgen_2d_true_optical_depth_comparison_v2_aggregate_v1_attempt_01/lightgen_2d_true_optical_depth_comparison_v2_aggregate_report.v1.json)，SHA-256=`78fc9061dacac1f68180b2d4619ca35cb3356b7c4b007cfe37f21a8092cfd002`；[lineage](../../outputs/20260914_n5_v2/runs/lightgen_2d_true_optical_depth_comparison_v2_aggregate_v1_attempt_01/lightgen_2d_true_optical_depth_comparison_v2_aggregate_lineage.v1.json)，SHA-256=`1662dabded7fda151bfb017c45b73de89cf75b745fd296c7b4074a770a8a8e67`；classification=`OPTIMIZATION_HORIZON_DOMINANT_TRUE_DEPTH_NOT_SUPPORTED`；Outer Gate=`N5_V2_TRUE_OPTICAL_DEPTH_COMPARISON_RESOLVED`。

关键结果：

| candidate | Smooth 0/100/200/300 | `R300-0.4838` | paired 95% CI | DEV image delta |
| --- | --- | ---: | --- | ---: |
| A continued C1 | 0.4838 / 0.5254 / 0.5294 / 0.5526 | +0.0688 | [+0.0612,+0.0768] | +0.00256892 |
| B true depth C1 | 0.4838 / 0.5240 / 0.5278 / 0.5504 | +0.0666 | [+0.0582,+0.0746] | +0.00249731 |

主比较 `Delta_depth=R_B300-R_A300=-0.0022`，paired bootstrap 95% CI=`[-0.0042,-0.0002]`；
预注册的 `Delta_depth>=+0.05` 与 CI lower `>0` 均不满足，故 V2 结论是固定 `300`-update
窗口内 `OPTIMIZATION_HORIZON_DOMINANT_TRUE_DEPTH_NOT_SUPPORTED`。两臂 DEV reconstruction、
cyclic template/Smooth transfer、raw validity 均通过。A 的 trainable count=`23,600`；B 的
trainable count=`33,600`，其中额外 `10,000` 为 midpoint depth phase。step-300 adapter
residual Fro 为 A=`5.39799668`、B=`5.40529312`；generator phase-change Fro 为 A=`12.30389625`、
B=`11.55147097`；B midpoint phase-change Fro=`7.82456298`。

Stage-A firewall 独立保持 `0.937`：A=`0.5526<0.937`、B=`0.5504<0.937`，均为
`STAGE_A_FIREWALL_FAILED`，`stage_b_authorized=false`。实际更新仅限声明的 adapter/generator
(B 含 midpoint phase)，encoder、OLS、`W_OLS` 未更新；没有 retry、rank/loss/lambda/lr 调整、
Stage B、generation、protected/test/sealed read 或 promotion。原 N4 aggregate 仍 unresolved，
V1 INVALID artifact 未被改写或消费。

## N4 OLS-interface/synthesis model-family comparison（2026-09-13/14，INVALID；C1 one-shot harness stop，C2 valid candidate only）

用户明确授权 `EXECUTE_N4_OLS_INTERFACE_SYNTHESIS_MODEL_FAMILY_COMPARISON_V1`。在
`research/lightgen-ols-native-semantic-generation-v1`、HEAD=`a9cd1749adfa690e282fd52a6425564fc2cf102c`
上完成 bounded preflight，以及 C1/C2 各自唯一一次 scientific scope。C0 沿用 V11 rank-9
endpoint；C1 为 rank-9 adapter + 可训练当前 2-layer generator；C2 为 rank-9 adapter +
当前 2-layer propagation stack + 可训练 co-planar entrance phase screen。encoder、OLS、
`W_OLS`、Smooth evaluator 与数据角色冻结；无 retry、调参、Stage B、generation、protected/
test/sealed read 或 promotion。

- authority=`DEFINE_N4_OLS_INTERFACE_SYNTHESIS_MODEL_FAMILY_CONTRACT_V1`；continuation contract [contract](../../outputs/20260913_n4/runs/lightgen_2d_ols_interface_synthesis_model_family_contract_v1_attempt_05/lightgen_2d_ols_interface_synthesis_model_family_contract.v1.json)，SHA-256=`ae3b90bb84e7748094217ff04ef8e9141c9dd1ab5e89a21376efc73978ee2cd3`。
- continuation preflight [report](../../outputs/20260913_n4/runs/lightgen_2d_ols_interface_synthesis_model_family_preflight_n4_attempt_05/lightgen_2d_n4_model_family_preflight_report.v1.json)，SHA-256=`3fec74d08145d5b368647d94cf9ceb69754a8f7c73abceed51f3ebbafc530510`；`AUTHORIZABLE`，scientific=`0`。
- C1 [report](../../outputs/20260913_n4/runs/lightgen_2d_ols_interface_synthesis_model_family_c1_attempt_01/lightgen_2d_n4_c1_scientific_report.v1.json)，SHA-256=`2b6b8c9d4b5bbe2c48eab18ee0d855c95f699cc0abeb93202bfe6340e51c6e61`；`INVALID/invalid_harness`，classification=`INVALID_N4_SCIENTIFIC_UNEXPECTED_EXECUTION_ERROR`。C1 完成 `300` updates，但末端 resume probe 请求耗尽 sampler 的下一批，报告未形成可接受的 C1 scientific metrics；没有 retry。
- C2 [report](../../outputs/20260913_n4/runs/lightgen_2d_ols_interface_synthesis_model_family_c2_attempt_01/lightgen_2d_n4_c2_scientific_report.v1.json)，SHA-256=`660454787c3df9d5b6aa69facfd9ec94acc67c19ac08cff1e23cdcf754b81003`；lineage SHA-256=`21c6840d62012a9fa7a9cca6ffd2cc377aa3be3e2832223f5aead1499763edbd`；`VALID_COMPLETED/N4_CANDIDATE_ADEQUATE`。

C2：Smooth `0.3056→0.4844`，delta=`+0.1788`，paired 95% CI=`[+0.16759,+0.18980]`；DEV
image `0.0531686321→0.0487769855`，delta=`-0.00439165`；cyclic template/Smooth transfer、
broad/not-pair-dominated 与 raw validity 均通过。C2 单臂满足 candidate criteria，但 C1
invalid 使对称 aggregate 无法成立；因此不得声称 N4 comparative Gate resolved，不正式
选择 C2。C2 Stage-A 仍失败：`0.4844<0.937`。

当前边界：N4 不开放 Stage B、generation、scale/prior、protected/test/sealed read 或
formal promotion。末端 sampler harness 修复已进入未提交 N4 脚本并由 continuation
preflight 验证，但不改写 C1 immutable report；没有重跑 C1，也没有执行新的 aggregate。

## N3 low-rank model-order selection（2026-09-12，VALID_COMPLETED；一次性候选执行）

用户明确授权 `授权执行 N3 model-order selection`。在
`research/lightgen-ols-native-semantic-generation-v1`、HEAD=`a9cd1749adfa690e282fd52a6425564fc2cf102c`
上完成 N3 contract、bounded preflight 与 rank=`1..9` 的 model-order selection：rank=`1..8`
各一次新的 `300`-update CUDA arm，rank=`9` 复用已核验等价的 V11 authority；未执行
rank=0、rank>9、rank sweep 之外的扩展、retry、second arm、调参或任何下游阶段。

- N3 semantic authority=`DEFINE_N3_LOW_RANK_MODEL_ORDER_CONTRACT_V1`；contract SHA-256=`d2b72ce978e4f28fdbb89441a0e84bc7e7792726def310beac129e94b9b4b181`。
- preflight attempt-01/02=`INVALID_HARNESS`（scientific=`0`）；修正 rank-specific gradient probe 后 attempt-03=`AUTHORIZABLE`，report SHA=`cb77e170cb3814533df9c73c1571cbd50e548527d82ed6ecf650ed8d32345af8`，lineage SHA=`7e9e1502455c62096f71f6dd4c0615c4c192f9e87e721afa6ca48173794b323e`。
- aggregate report SHA=`1b6bd6af18462f717fa19532e1a981e872a4e48fe6498ed561890f6570ae89de`，lineage SHA=`93c3d4ffcfad5e68b435d7063e6e6c834b700cde2e4b3ccd95391f3bee8b8a8f`；classification/Outer Gate=`N3_LOW_RANK_MODEL_ORDER_RESOLVED`。

冻结 V11 条件：`T_r=I_100+U_rV_r^H`、nested `U0=0.01*[e0..e_(r-1)]`、`V0=0`，
semantic/image objective、`lambda_image=13.353282014834436`、encoder/generator/OLS/
`W_OLS`、数据角色、optimizer、seed、batch 与 budget 全部保持固定，仅 adapter U/V
可训练。首次 controller 在 rank-1 完整有效报告之后发生 assembly read bug；修正后
消费该已有报告并继续 rank=`2..8`，没有重跑 rank-1。

| rank | R300 | Delta Smooth | CI lower | DEV image Delta | cyclic | adequacy |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 0.2464 | +0.1428 | +0.13559 | -0.011420 | 10/10 broad | pass |
| 2 | 0.2504 | +0.1468 | +0.13960 | -0.011015 | 10/10 broad | pass |
| 3 | 0.2518 | +0.1482 | +0.14100 | -0.010788 | 10/10 broad | pass |
| 4 | 0.2534 | +0.1498 | +0.14219 | -0.011470 | 10/10 broad | pass |
| 5 | 0.2572 | +0.1536 | +0.14600 | -0.011971 | 10/10 broad | pass |
| 6 | 0.2614 | +0.1578 | +0.15000 | -0.012548 | 10/10 broad | pass |
| 7 | 0.2668 | +0.1632 | +0.15540 | -0.013109 | 10/10 broad | pass |
| 8 | 0.2760 | +0.1724 | +0.16420 | -0.013594 | 10/10 broad | pass |
| 9（V11 reuse） | 0.3056 | +0.2020 | +0.19360 | -0.013893 | 10/10 broad | pass |

全部 rank=`1..9` 均满足候选 adequacy；按最低可行阶规则选择 `r*=1`，不是按最大
Smooth 或复杂度惩罚选择。rank-9 的 residual Fro step-300=`3.02419`，selected rank-1
为 `2.04873`；完整 singular-value 向量保存在 aggregate/candidate artifacts 中。

Stage-A firewall 仍独立失败：selected rank-1 `R300=0.2464<0.937`，rank-9 也仅
`0.3056<0.937`。因此 `N3_LOW_RANK_MODEL_ORDER_RESOLVED` 不授权 Stage B、generation、
scale/prior、protected/test/sealed read 或 formal promotion；generation readiness 仍
`not_ready`。

## V11 low-rank semantic/reconstruction compatibility（2026-09-12，VALID_COMPLETED；一次且仅一次）

用户明确授权 `EXECUTE_LIGHTGEN_2D_LOW_RANK_SEMANTIC_RECONSTRUCTION_COMPATIBILITY_V11`。
在 `research/lightgen-ols-native-semantic-generation-v1`、HEAD=`a9cd1749adfa690e282fd52a6425564fc2cf102c`
上完成 V11 READY contract、`AUTHORIZABLE` bounded preflight 与唯一一次 `300`-update CUDA
scientific arm；没有 retry、second arm、rank sweep、lambda/lr tuning 或下游阶段。

- contract SHA-256=`6c49a0cff4e08622351438529cebafe9388537d713a5b78b9fabbc953904efff`；preflight report SHA=`b5078cdd61664f447ab880ad32f57b6f0d4287ea88f446c111414560f6743386`，lineage SHA=`4910a4de5b928f97c4c1ecf9af178ad66e3cf54ad94e22f5899a4f3f9fe4f695`。
- scientific input SHA=`b0c5c6a80f0cd58be385762f5bf17b703e0f790a76aefb91586705874601b37a`；report SHA=`bc603b7a3b7ef1590bc354cb0b62464f3ececcb2390834d864d58bfbe9158184`；lineage SHA=`2c7d7c99a48c6a5a60d6524a38aa9facd6b7e0203138689f3830dcc7308ac9c8`。
- objective：`L_V11=L_sem+lambda_image*L_image`，`L_sem=L_anchor+28.48464149920509*L_transfer`；same initial TRAIN batch gave `g_sem=0.0016925142105153586`、`g_image=0.00012674892923566114`、frozen `lambda_image=13.353282014834436`。V11 factor width/rank=`9`，`U0=0.01*I_9`、`V0=0`；only adapter 3,600 real parameters trainable，encoder/generator/OLS/`W_OLS`/prototypes/templates/Smooth frozen。

| step | DEV L_image | Smooth macro | template-anchor macro | correct-template margin |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.06706161 | 0.1036 | 0.1706 | -0.00810254 |
| 100 | 0.05742796 | 0.2488 | 0.5156 | 0.00154549 |
| 200 | 0.05274165 | 0.2478 | 0.5530 | 0.00209791 |
| 300 | 0.05316863 | 0.3056 | 0.5980 | 0.00338574 |

V11 Outer Gate=`V11_LOW_RANK_SEMANTIC_RECONSTRUCTION_COMPATIBILITY_SUPPORTED`：
Smooth `Delta_R=+0.2020`，paired 95% CI=`[+0.1936,+0.2102]`；DEV image delta=`-0.013893`
（阈值 `<=+0.005`）；known cyclic template/Smooth transfer 均 retained，且均 `10/10`
positive pair、raw validity=`true`。step-300 `UV^H` residual Fro=`3.02419035`、spectral=`2.67642540`；前 9 个 singular values=`2.67642540,1.40705576,0.05017205,0.01090476,0.00563522,0.00061744,0.00037506,0.00014226,0.00008291`，理论 residual rank=`9`，numerical rank=`100` 仅是 float32 noise 诊断。

Stage-A 独立 firewall 仍失败：`R300=0.3056 < 0.937`，`STAGE_A_FIREWALL_FAILED`。因此 Stage B、generation、scale/prior、protected/test/sealed、formal promotion 均关闭；本次确认 adapter update=`YES`，encoder/generator update=`NO`，rank sweep/retry/second arm=`NO`。

报告保持 write-once；执行后仅修正了 V11 executor 的 future report-assembly 元数据覆盖，不改写本次 report/lineage，不构成 scientific retry 或新证据。

## V10.1 low-rank OLS-to-generator remap canary（2026-09-12，VALID_COMPLETED；唯一一次）

V10.1 authority `DEFINE_LIGHTGEN_2D_LOW_RANK_OLS_TO_GENERATOR_REMAP_CANARY_CONTRACT_V10_1` 仅修正旧 V10 不可能的 step-0 初始化：`T_adapt=I_100+UV^H`、factor width=`9`、`U0=0.01*I_9`、`V0=0`；其余 V10 scientific semantics、V9 anchor+transfer objective、数据角色、Outer Gate 与 Stage-A firewall 保持不变。旧 V10 attempt-01/02 未改、未重跑。

- contract SHA-256=`45210d92e00ba01255a4340dfc837c514c110640097aa20821d8d5b6e0efab3d`；preflight=`AUTHORIZABLE`，report SHA=`e651e51a3d022b4ec041107e313a6e774a8740b7cd9dbd34f6a50c57ef670bf0`，lineage SHA=`1eb2508d6a8965d78ed4070b508718b2cdb69084bc3d5ff4bdeeec84cd3cf5d1`。
- preflight：`||U0||F=0.03`、`||V0||F=0`、`T0=I`；production synthetic canonical/identity-adapter raw allclose=`true`；`grad_U=0` allowed，anchor `grad_V=0.0010775054`，combined total=`0.0016925142`；one-step `||T-I||F=0.0004217633`；CUDA smoke、checkpoint/resume、exact-core synthetic=`PASS`。
- scientific report SHA-256=`0eb96990db673c82526c2ea37bd15afadb9094870eb0afe3ff7af84788b6b4212`；lineage SHA-256=`4b835f4fd7cae36750a7e28ba324cb07453440d68ee12187982db85a0704766f`；唯一一次 `300` updates，classification=`V10_LOW_RANK_OLS_TO_GENERATOR_REMAP_SUPPORTED`。
- Smooth `R0=0.1036`、`R300=0.3336`、`Delta_R=+0.2300`、paired 95% CI=`[+0.22099,+0.23960]`；cyclic transfer retained=`true`，10/10 pair positive，raw validity=`true`。DEV image loss `0.0670616→0.2914308`，说明语义提升伴随图像重建恶化。
- Stage-A 独立 firewall：`R300=0.3336 < 0.937`，`STAGE_A_FIREWALL_FAILED`；Stage B、generation、scale/prior、protected/test/sealed、formal promotion 均关闭。adapter residual step-0=`0`；step-300 Fro=`6.42947`、spectral=`6.06980`；默认 numerical rank=`100` 是 float32 identity-subtraction noise 下的诊断值，factor width/theoretical residual rank 仍为 `9`。
- 边界：branch/HEAD start=end=`research/lightgen-ols-native-semantic-generation-v1` / `a9cd1749adfa690e282fd52a6425564fc2cf102c`，worktree status unchanged；adapter update=`YES` only，encoder/generator/OLS/W_OLS update=`NO`，retry/second arm/rank sweep=`NO`，protected/test/sealed=`NO`，Stage B/generation/promotion=`NO`。

## V10 low-rank OLS-to-generator remap canary（2026-09-12，INVALID；one-shot preflight stop）

用户明确授权 `EXECUTE_LIGHTGEN_2D_LOW_RANK_OLS_TO_GENERATOR_REMAP_CANARY_V10`。在
`research/lightgen-ols-native-semantic-generation-v1`、HEAD=`a9cd1749adfa690e282fd52a6425564fc2cf102c` 上完成 V10 contract 物化；唯一一次 bounded preflight 在 scientific forward 前以 `INVALID_HARNESS` 停止。授权明确禁止 retry，因此没有重跑 preflight，也没有进入 300-step scientific training、DEV 或 cyclic scientific evaluation。

- [V10 contract](../../outputs/20260912_v10/runs/lightgen_2d_low_rank_ols_to_generator_remap_canary_contract_v10_attempt_01/lightgen_2d_low_rank_ols_to_generator_remap_canary_contract.v10.json)，SHA-256=`5efeb9d4b8e5e1e29fee3e3c445a52a201fbfabdd7f696a183284c71832ca0f1`，READY/valid_completed；semantic authority=`DEFINE_LIGHTGEN_2D_LOW_RANK_OLS_TO_GENERATOR_REMAP_CANARY_CONTRACT_V10`，父 authority 为 V9。
- [preflight report](../../outputs/20260912_v10/runs/lightgen_2d_low_rank_ols_to_generator_remap_canary_preflight_v10_attempt_01/lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_preflight_report.v1.json)，SHA-256=`c0d1cf193de37c7ed40e0d979769ae93dc3c08193d57e43955a1260fecb25f00`；[lineage](../../outputs/20260912_v10/runs/lightgen_2d_low_rank_ols_to_generator_remap_canary_preflight_v10_attempt_01/lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_preflight_lineage.v1.json)，SHA-256=`fc3585925c0720ef532df8251c660ef6cb47b249360139020c464014b249458f`；classification=`INVALID_V10_PREFLIGHT_HARNESS`。

失败原因为入口 harness 的参数名 double-prefix：firewall probe 将已有 `adapter.` 前缀再次添加，导致 `V10 combined probe u_real gradient missing`；同时已修正 complex gradient norm helper 的 imaginary 分量处理。修复未再次执行，故不产生科学结论。attempt-01 事实边界：scientific forward/update=`0`、branch/HEAD 未变、worktree status 未变、protected/test/sealed=`0`、retry=`NO`。V10 Outer Gate 与 Stage-A eligibility unresolved；Stage B、generation、formal promotion 未开放。

## V10 low-rank OLS-to-generator remap canary attempt-02（2026-09-12，INVALID；one-shot preflight stop）

用户授权覆盖修复后的 V10 engineering/preflight attempt-02，并规定仅当 preflight=`AUTHORIZABLE` 才进入唯一一次 scientific run。实际在同一 READY contract、branch=`research/lightgen-ols-native-semantic-generation-v1`、HEAD=`a9cd1749adfa690e282fd52a6425564fc2cf102c` 上执行后，入口在 scientific forward 前的 step-0 identity probe 以 `INVALID_V10_PREFLIGHT_HARNESS` 停止；因此没有进入 scientific training、DEV 或 cyclic evaluation，也没有重试。

- [preflight report](../../outputs/20260912_v10/runs/lightgen_2d_low_rank_ols_to_generator_remap_canary_preflight_v10_attempt_02/lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_preflight_report.v1.json)，SHA-256=`cf50ae067c28de08b9bf27eb88294b34998842b84833eaa464eb27a7ab119d74`；[lineage](../../outputs/20260912_v10/runs/lightgen_2d_low_rank_ols_to_generator_remap_canary_preflight_v10_attempt_02/lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_preflight_lineage.v1.json)，SHA-256=`b599330fe454ea31d466a9c657b205ac6b530eeb1ae67f57215c7ac23bb746e4`。
- 失败原因：`V10 step-0 adapter changed encoded coefficients`。rank-9 adapter 的 identity-preserving initialization 未通过 frozen tolerance；故不能把 attempt-01 的 double-prefix 与 complex-gradient-norm 修复视为已通过，也不产生 V10 scientific classification。
- attempt-02 边界：scientific forward=`0`、scientific updates=`0`、scientific checkpoint=`0`、protected/official-test/sealed=`0`；branch、HEAD、worktree status 未改变。V10 Outer Gate 与 Stage-A eligibility unresolved；Stage B、generation、formal promotion 继续关闭。

## V10 step-0 identity engineering stabilization（2026-09-12，BLOCKED；冻结合同代数不一致）

在不读取 scientific TRAIN/DEV、不创建 attempt-03、且不修改 attempt-01/02 的前提下，对实际生产 `LowRankAdapter` 做了只读 algebraic/synthetic identity audit。attempt-02 probe 比较同一批 `encoded_result.z_ols.coefficients` 与实际 `adapter(z)` 输出；形状和复数表示一致，故不是不同表示或 port-order 混比。

冻结合同的 `U=s[e_0,...,e_8]`、`V=s[e_9,...,e_17]`、`s=0.01` 与 `T=I+UV^H` 在当前参数化下给出 9 个 `1e-4` 非零残差，`||UV^H||_F=3e-4`，实测 `max_abs(T-I)=1e-4`，identity allclose 失败。所谓“orthogonal-support”是行支持，不会令同列配对的 `UV^H` 为零；修复必须改变初始化/参数化或有效列支持，超出本任务范围。

因此 exact step-0 identity 与冻结的 rank-9 双非零因子初始化不能同时成立；identity tolerance、adapter、rank、objective、lambda 和 contract 均未修改。本稳定化任务为 `BLOCKED`，V10 attempt-03 未执行，scientific updates=`0`，V10 Outer Gate 仍 unresolved。

## V9 joint semantic remapping warmup scientific execution（2026-09-12，VALID_COMPLETED；一次且仅一次）

用户明确授权 `EXECUTE_LIGHTGEN_2D_JOINT_SEMANTIC_REMAPPING_WARMUP_V9`。在
`research/lightgen-ols-native-semantic-generation-v1` 上，start/end HEAD 均为
`a9cd1749adfa690e282fd52a6425564fc2cf102c`；工作区既有 dirty 状态保持，scientific
provenance 的 `worktree_status_unchanged=true`。V9 contract 已 READY，bounded preflight
为 `AUTHORIZABLE`，随后完成且仅完成一次 300-step CUDA joint scientific arm；没有 retry、第二
arm、超预算、Stage B、generation 或 formal promotion。

- [V9 contract](../../outputs/20260912_v9/runs/lightgen_2d_joint_semantic_remapping_warmup_contract_v9_attempt_01/lightgen_2d_joint_semantic_remapping_warmup_contract.v9.json)，SHA-256=`4d0fb81bfa6152995aad4484c83487c20335616283097b394757f4616bf41fca`；semantic authority=`DEFINE_LIGHTGEN_2D_JOINT_SEMANTIC_REMAPPING_WARMUP_CONTRACT_V9`，父合同为 V8。
- preflight [report](../../outputs/20260912_v9/runs/lightgen_2d_joint_semantic_remapping_warmup_preflight_v9_attempt_01/lightgen_2d_joint_semantic_remapping_warmup_v9_preflight_report.v1.json)，SHA-256=`8baa04ac089ff82aebf3fefb9de9ff70a0d495022b5c19e852ab8eb82926e2f8`；[lineage](../../outputs/20260912_v9/runs/lightgen_2d_joint_semantic_remapping_warmup_preflight_v9_attempt_01/lightgen_2d_joint_semantic_remapping_warmup_v9_preflight_lineage.v1.json)，SHA-256=`002fd2d5303682f30f6b0b54f580acd99f6bd8456d2d55c441215855e0e7fb7c`；`scientific_300_step_executed=false`。
- [scientific input](../../outputs/20260912_v9/runs/lightgen_2d_joint_semantic_remapping_warmup_v9_scientific_attempt_01/lightgen_2d_joint_semantic_remapping_warmup_v9_scientific_input.v1.json)，SHA-256=`6cfedfb607834bd99784f7082535628c727436c9006216e67781afbe326b4345`；[report](../../outputs/20260912_v9/runs/lightgen_2d_joint_semantic_remapping_warmup_v9_scientific_attempt_01/lightgen_2d_joint_semantic_remapping_warmup_v9_scientific_report.v1.json)，SHA-256=`280cd33abbe3195856798989d7f300c9fcc618e57906149c3abc0abdfd3d7465`；[lineage](../../outputs/20260912_v9/runs/lightgen_2d_joint_semantic_remapping_warmup_v9_scientific_attempt_01/lightgen_2d_joint_semantic_remapping_warmup_v9_scientific_lineage.v1.json)，SHA-256=`47d8768165b5d5207df8a190547b56223b49f4a9cf89e7da9158be43f05fa418`。
- [sufficient observations](../../outputs/20260912_v9/runs/lightgen_2d_joint_semantic_remapping_warmup_v9_scientific_attempt_01/lightgen_2d_joint_semantic_remapping_warmup_v9_sufficient_observations.v1.pt)，SHA-256=`48cff565de6144d941715dd497c47c6fccc956e6e0117ad896b49e76b82f33ff`；[cyclic raw](../../outputs/20260912_v9/runs/lightgen_2d_joint_semantic_remapping_warmup_v9_scientific_attempt_01/lightgen_2d_joint_semantic_remapping_warmup_v9_step300_cyclic_raw_outputs.v1.pt)，SHA-256=`8556a1e8db945d73cf4dab680348b5b7eec8af737655d5561828a7a416c1784f`；sufficient/raw/report/lineage reload=`PASS`。

V9 与 V8 只改变一个主自由度：初始状态仍为 exact V3 CONTROL encoder + exact V4 valid generator，但 `encoder.phase` 和 `generator.phase` 均 trainable（各 `20,000`，合计 `40,000`）；OLS、`T=I_100`、`W_OLS`、prototypes、templates、Smooth v2.1 与其余 runtime 全冻结。目标仍为
`L_V9=L_anchor+28.48464149920509*L_transfer`，`s_tau=0.02919053753794475`；image loss、prototype loss 与 Smooth/Frozen evaluator 均没有训练梯度。设备为 RTX 3060 Laptop、compute capability `8.6`、PyTorch `2.5.1`、CUDA `12.1`。

| step | TRAIN L_anchor | TRAIN L_transfer | TRAIN L_total | TRAIN L_image | DEV L_image | Smooth macro | anchor macro | anchor margin | OLS margin |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.22846563 | 0.69342184 | 21.98033819 | 0.07510780 | 0.06706161 | 0.1036 | 0.1706 | -0.00810254 | 0.00208249 |
| 100 | 2.20848503 | 0.69193470 | 21.91799689 | 0.06499042 | 0.06625223 | 0.1054 | 0.2050 | -0.00670630 | 0.00206976 |
| 200 | 2.20736347 | 0.68992402 | 21.85960196 | 0.06401648 | 0.06622797 | 0.1100 | 0.2386 | -0.00568535 | 0.00204236 |
| 300 | 2.16701212 | 0.68808337 | 21.76682022 | 0.06836181 | 0.06666521 | 0.1184 | 0.2670 | -0.00486589 | 0.00200970 |

自然 Smooth `Delta_R=+0.014800`，paired 95% CI=`[+0.011400,+0.018610]`：CI lower 为正，
但预注册 `Delta_R>=+0.05` 未通过。template-anchor macro=`0.1706→0.2670`，correct-class
margin=`-0.00810254→-0.00486589`；OLS margin Delta=`-0.0000727933`；这些均是描述性机制指标。

已知 cyclic transfer retained=`true`：template mean/median=`0.01088561/0.00826996`，CI
`[0.01061948,0.01115307]`；Smooth mean/median=`3.10276725/2.93491745`，CI
`[2.99842256,3.20315371]`；10/10 positive pairs，`broad=true`、`pair_dominated=false`。
十个 pair 的 template/Smooth means 为：

| pair | template | Smooth |
| --- | ---: | ---: |
| 0→1 | 0.02981124 | 4.79306646 |
| 1→2 | 0.01347965 | 4.17880627 |
| 2→3 | 0.00287232 | 0.27432058 |
| 3→4 | 0.01103936 | 5.93916874 |
| 4→5 | 0.00493736 | 2.87124777 |
| 5→6 | 0.00494553 | 1.57296430 |
| 6→7 | 0.00854289 | 3.56916991 |
| 7→8 | 0.00844004 | 2.67373318 |
| 8→9 | 0.00521505 | 1.36521842 |
| 9→0 | 0.01957267 | 3.78997682 |

所有 checkpoint 与 cyclic raw 均 finite/shape-valid/structured/nonconstant；step-0/300 exact
duplicate=`0`、unique=`5000`。因此 V9 primary criteria 为：natural Smooth=`FAIL`、known
cyclic transfer=`PASS`、raw validity=`PASS`，最终分类为
`V9_JOINT_SEMANTIC_REMAPPING_NOT_SUPPORTED`，Outer Gate
`V9_JOINT_SEMANTIC_REMAPPING_SUPPORTED` 未通过。该结果只支持小幅正向 bounded evidence，
不构成 Stage-A semantic eligibility。

Stage-A firewall 独立仍为 `R300=0.1184 < 0.937`，`STAGE_A_FIREWALL_FAILED`。本次确认
encoder update=`YES`、generator update=`YES`、L_image/prototype/Smooth gradient=`NO`；
retry、Stage B、generation、protected/test/sealed、formal promotion 全部=`NO`。下一步不由
该结果自动开放任何下游执行。

## V8 absolute-anchor plus transfer warmup scientific execution（2026-09-12，VALID_COMPLETED；一次且仅一次）

用户已明确授权 `EXECUTE_LIGHTGEN_2D_ABSOLUTE_ANCHOR_PLUS_TRANSFER_WARMUP_V8`。在
`research/lightgen-ols-native-semantic-generation-v1`、start/end HEAD=`a9cd1749adfa690e282fd52a6425564fc2cf102c` 上完成 V8 READY contract、一次 `AUTHORIZABLE` engineering preflight 与唯一一次 300-step CUDA scientific arm；worktree 原有 dirty 状态、branch、HEAD 和 status line 均保持不变。

结果=`VALID_COMPLETED/valid_completed`，唯一科学分类=`V8_ABSOLUTE_ANCHOR_PLUS_TRANSFER_NOT_SUPPORTED`；Outer Gate `V8_ABSOLUTE_ANCHOR_PLUS_TRANSFER_SUPPORTED` 未满足。preflight [report](../../outputs/20260911_v8/runs/lightgen_2d_absolute_anchor_plus_transfer_warmup_preflight_v8_attempt_01/lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_preflight_report.v1.json) SHA-256=`e197bff80ea4fbb42d2a0c141028ecf7cabe34416d5816df6a95620cc9a3faad`、[lineage](../../outputs/20260911_v8/runs/lightgen_2d_absolute_anchor_plus_transfer_warmup_preflight_v8_attempt_01/lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_preflight_lineage.v1.json) SHA-256=`c2dcd73581143d288b5e536252f7188cfeabb82c624d6756444ec868fbf9b2c2`，均 `AUTHORIZABLE`/reload 通过且 `scientific_300_step_executed=false`。

- [V8 contract](../../outputs/20260911_v8/runs/lightgen_2d_absolute_anchor_plus_transfer_warmup_contract_v8_attempt_01/lightgen_2d_absolute_anchor_plus_transfer_warmup_contract.v8.json)，SHA-256=`02cef2918ff89eafaa53cf52f7a7492b3091e84acba3c351b6142b94b703a85d`，READY/valid_completed。
- [scientific input](../../outputs/20260911_v8/runs/lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_scientific_attempt_01/lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_scientific_input.v1.json)，SHA-256=`4942c79d1ed4df92ddbee5cfbcb9771e749750c3ceeaffac652869c2e4b8673d`；[report](../../outputs/20260911_v8/runs/lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_scientific_attempt_01/lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_scientific_report.v1.json)，SHA-256=`5d534f9f7e7192c807a3c927a0815680c17b5d00d7f9edaf8c9987cf3bd39806`；[lineage](../../outputs/20260911_v8/runs/lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_scientific_attempt_01/lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_scientific_lineage.v1.json)，SHA-256=`69cfcf2e34e02d192c1f7cbca5f98bc960c163371a28404d3fc0fff9a0fad121`。
- [sufficient observations](../../outputs/20260911_v8/runs/lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_scientific_attempt_01/lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_sufficient_observations.v1.pt)，SHA-256=`dd3848bdcf85c85d06eb0ca220b9f4149ff8df52701989416aa563bb6441a415`；[cyclic raw outputs](../../outputs/20260911_v8/runs/lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_scientific_attempt_01/lightgen_2d_absolute_anchor_plus_transfer_warmup_v8_step300_cyclic_raw_outputs.v1.pt)，SHA-256=`9599e0f52bfef49cbf74c615a39be12460454cf16c65e4a1c115049fe4814ce5`；reload 均通过。

冻结 objective=`L_V8=L_anchor+lambda_transfer*L_transfer`，仅 `encoder.phase` 训练；V3 CONTROL encoder、V4 generator、100-port complex OLS、`T=I_100`、`W_OLS`、V5 TRAIN prototypes、历史 TRAIN templates 与 Smooth evaluator 冻结。`s_tau=0.02919053753794475`；初始 `g_anchor=0.0082287618885706`、`g_transfer=0.00028888416448265975`，固定 `lambda_transfer=28.48464149920509`，无裁剪/调参/重算。preflight 覆盖真实 3-step CUDA、encoder-only/frozen/generator/Smooth firewall、negative controls、no-data smoke、checkpoint/reload/resume 与 exact-core synthetic；未访问 protected/test/sealed。

| step | TRAIN L_anchor | TRAIN L_transfer | TRAIN L_total | DEV L_image | Smooth macro | OLS margin | anchor macro |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.22846563 | 0.69342184 | 21.98033819 | 0.06706161 | 0.1036 | 0.00208249 | 0.1706 |
| 100 | 2.23859913 | 0.69290392 | 21.97571897 | 0.06671077 | 0.1040 | 0.00206943 | 0.1826 |
| 200 | 2.24820785 | 0.69156806 | 21.94727610 | 0.06650268 | 0.1062 | 0.00203811 | 0.1914 |
| 300 | 2.22151052 | 0.69053010 | 21.89101275 | 0.06643724 | 0.1086 | 0.00199363 | 0.2022 |

Natural Smooth `Delta_R=+0.005000`，95% CI=`[+0.002800,+0.007400]`，但未达到 `+0.05`；cyclic transfer retained=true，template mean=`0.00583127524` CI lower=`0.00562985606`，Smooth mean=`1.94132099` CI lower=`1.83938933`，positive pairs=`10/10`，broad=true，pair-dominated=false。anchor macro 为描述性指标，不构成额外 Gate；raw validity 全通过。Stage-A firewall 仍为 `R300=0.1086 < 0.937`，故 Stage B 不开放。

FACT：该结果只支持在固定 optical/runtime、预算与 frozen authorities 下的 bounded V8 outcome；MODEL_ASSUMPTION：absolute image anchor 与 transfer coupling 可能改善 natural semantic reconstruction；DESIGN_DECISION：anchor 使用历史 TRAIN templates 与冻结 `s_tau`，`lambda_transfer` 按初始首批梯度一次性配平，step-0 为 matched baseline。没有 retry、second arm、Stage B、scale/prior/generation、protected/test/sealed read 或 formal promotion；`V5_LATENT_TO_OPTICAL_TRANSFER_MECHANISM_RESOLVED` 仍 unresolved。

## V5 class-direction transfer operating-point stability diagnosis（2026-09-11，VALID_COMPLETED；一次且仅一次）

按明确授权 `DIAGNOSE_V5_CLASS_DIRECTION_TRANSFER_OPERATING_POINT_STABILITY_V1`，在 `research/lightgen-ols-native-semantic-generation-v1`、start/end HEAD=`a9cd1749adfa690e282fd52a6425564fc2cf102c` 上完成一次 forward-only/no-update diagnosis。历史有效 authority `OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_SUPPORTED`、冻结 V5 CONTROL/CANDIDATE step-300 checkpoints、V5 选择索引与固定 cyclic rule `t=(y+1) mod 10` 均通过验证；未执行 optimizer、backward、parameter/optical update、V5 retry、Stage B、generation 或 protected/test/sealed read。

结果=`VALID_COMPLETED/valid_completed`，唯一分类=`V5_CONTROL_OPERATING_POINT_TRANSFER_LOSS_SUPPORTED`；Outer Gate `V5_CLASS_DIRECTION_TRANSFER_OPERATING_POINT_STABILITY_RESOLVED` 已满足。该结果只证明本次固定操作点的 bounded transfer-loss stability，不改变 V5 latent-to-optical transfer-coupling 的 unresolved 状态，也不构成训练、Stage B、generation 或 formal promotion authorization。

- [`input`](../../outputs/20260911_v5/runs/lightgen_2d_v5_class_direction_transfer_operating_point_stability_v1_attempt_01/lightgen_2d_v5_class_direction_transfer_operating_point_stability_input.v1.json)，SHA-256=`2d4dce07d24b8e6b32a37e3570c6ec1bdc6ab1b2f18dc11999cf434ef7e85aed`；[`report`](../../outputs/20260911_v5/runs/lightgen_2d_v5_class_direction_transfer_operating_point_stability_v1_attempt_01/lightgen_2d_v5_class_direction_transfer_operating_point_stability_report.v1.json)，SHA-256=`1d8e0e232d9a5ef05fa964688c11847d960466d6b7e50cef71d7162b7d953ee`；
- [`lineage`](../../outputs/20260911_v5/runs/lightgen_2d_v5_class_direction_transfer_operating_point_stability_v1_attempt_01/lightgen_2d_v5_class_direction_transfer_operating_point_stability_lineage.v1.json)，SHA-256=`67f1b77594e944b5313b82f72574dde7ff21d2da54c3bcd97efd7386f34fea20`；sufficient observations SHA-256=`7d0281d66426906a648fc8d9aa284e563b6492cdd3011b06ddfb68db84c93a25`；attempt-03 audit artifact SHA-256=`c137313ecc5c3a0f8056a5be213bf5d4b1e1565316ecb5eeeb85aa5f7f2f0612`，audit=`PASS`。
- scientific forward count=`600`；Smooth evaluations=`400`；selection=`1000/100`，seed=`4321`；paired bootstrap=`B=999, seed=271828`，raw/sufficient/bootstrap reload 均为 `PASS`。

| operating point | template Δ mean [95% CI] | Smooth Δ mean [95% CI] | template transfer | Smooth transfer |
| --- | ---: | ---: | --- | --- |
| CONTROL | `-0.00099143 [-0.00113745, -0.00084232]` | `-0.05296442 [-0.15229365, 0.03803239]` | `false` | `false` |
| CANDIDATE | `-0.00127322 [-0.00142480, -0.00111903]` | `-0.09730077 [-0.19547855, -0.00736787]` | `false` | `false` |

两操作点 raw endpoint 均 finite、shape-valid、structured、nonconstant；本次只记录 bounded operating-point stability，未开放训练、Stage B、generation 或 formal promotion。

## V6 proto-only semantic warmup scientific execution（2026-09-11，VALID_COMPLETED；一次且仅一次）

按明确授权 EXECUTE_LIGHTGEN_2D_PROTO_ONLY_SEMANTIC_WARMUP_V6，在
research/lightgen-ols-native-semantic-generation-v1、start/end HEAD=a9cd1749adfa690e282fd52a6425564fc2cf102c 上完成 bounded preflight 后执行唯一一次 300-step scientific arm。preflight attempt-01/02 均在 scientific forward 前因 harness 问题停止并保持 immutable；attempt-03=AUTHORIZABLE 后未重跑 scientific arm。工作区原有 dirty 状态保持不变，branch/HEAD 未改变。

- 冻结 [V6 contract](../../outputs/20260911_v6/runs/lightgen_2d_proto_only_semantic_warmup_contract_v6_attempt_01/lightgen_2d_proto_only_semantic_warmup_contract.v6.json)，SHA-256=4b849632b67bd68f4075816e061bb3db0664abe31ab757a7137bfcd4c8f9f441，READY/valid_completed。
- preflight attempt-03 [report](../../outputs/20260911_v6/runs/lightgen_2d_proto_only_semantic_warmup_preflight_v6_attempt_03/lightgen_2d_proto_only_semantic_warmup_v6_preflight_report.v1.json)，SHA-256=d45082a1384f1001a63b69972203bf99d4d9095243c6ef5217f6d68a258559be；[lineage](../../outputs/20260911_v6/runs/lightgen_2d_proto_only_semantic_warmup_preflight_v6_attempt_03/lightgen_2d_proto_only_semantic_warmup_v6_preflight_lineage.v1.json)，SHA-256=15fef1def813b9ea6fd3f5dd33bf65c76e56fd7a782dc67b16a629474050054e；scientific_300_step_executed=false。
- scientific input SHA-256=c27d02af5acfd684136a01e857123c2d58872072ea0bb27cb238fb3be57b9550；[report](../../outputs/20260911_v6/runs/lightgen_2d_proto_only_semantic_warmup_v6_scientific_attempt_01/lightgen_2d_proto_only_semantic_warmup_v6_scientific_report.v1.json)，SHA-256=67918a6fe1dc4200b389a0ca2c4b419a6c150fb2f11288c1adc00e72605d09e9；[lineage](../../outputs/20260911_v6/runs/lightgen_2d_proto_only_semantic_warmup_v6_scientific_attempt_01/lightgen_2d_proto_only_semantic_warmup_v6_scientific_lineage.v1.json)，SHA-256=40153c45103e870c9ee3188feab18e141bfae31f18b98c611f21b6e8a5be1ac1。

冻结初值与配置：V3 CONTROL step-300 encoder、V4 valid step-300 generator、V5 TRAIN-only prototypes、s_mu=0.00555040966719389；仅 encoder.phase 可训练，generator/OLS/Smooth/prototypes 冻结；Adam lr=1e-3、batch=10、300 updates；L_V6=L_proto，image MSE 仅 evaluation-only；DEV=5000 rows，step=0/100/200/300；cyclic=1000/100，seed=4321，t=(y+1) mod 10。

| step | evaluation L_image | Smooth macro recall | OLS margin mean |
| ---: | ---: | ---: | ---: |
| 0 | 0.06706161 | 0.1036 | 0.00208249 |
| 100 | 0.06742571 | 0.1030 | 0.00211740 |
| 200 | 0.06777967 | 0.1030 | 0.00214080 |
| 300 | 0.06808599 | 0.1030 | 0.00215925 |

step-300 相对 step-0：Delta_L_image=+0.0010243780；OLS margin Delta=+7.67765e-5，95% CI=[+7.32591e-5,+8.00236e-5]；Smooth recall Delta=-0.0005960，95% CI=[-0.0014,0.0]。cyclic template Delta=-0.00172974 [-0.00188013,-0.00157520]、Smooth Delta=-0.17172088 [-0.26391294,-0.07869869]，两者 pass 均为 false；positive pairs 为 template 1/10、Smooth 4/10，known_cyclic_transfer_retained=false。

结果=VALID_COMPLETED/valid_completed，唯一分类=V6_PROTO_ONLY_SEMANTIC_WARMUP_NOT_SUPPORTED；Outer Gate V6_PROTO_ONLY_SEMANTIC_WARMUP_SUPPORTED 未满足。raw_validity_pass=true、ols_mechanism_pass=true，但 natural_smooth_pass=false、cyclic transfer 未保留。Stage-A firewall 独立为 R300=0.103 < 0.937，状态=STAGE_A_FIREWALL_FAILED；Stage B、generation、formal promotion 均关闭。

报告记录 training_authorized=false（仅表示该 bounded one-shot 已结束，不产生后续训练授权）、retry=false、generator_update=false、optical_update=false、protected=false、sealed=false。未执行 retry、第二 arm、scale/prior/generation、架构/OLS/W_OLS 修改或 protected/test/sealed read。V5 latent-to-optical transfer-coupling mechanism 仍 unresolved。

## V7 transfer-aware semantic warmup scientific execution（2026-09-11，VALID_COMPLETED；一次且仅一次）

按明确授权 EXECUTE_LIGHTGEN_2D_TRANSFER_AWARE_SEMANTIC_WARMUP_V7，在
research/lightgen-ols-native-semantic-generation-v1、start/end HEAD=a9cd1749adfa690e282fd52a6425564fc2cf102c 上完成 contract、bounded preflight 与唯一一次 300-step scientific arm。preflight attempt-01/02 均在 scientific forward 前因 synthetic harness 错误停止并保持 immutable；attempt-03=AUTHORIZABLE 后未重跑 scientific。工作区原有 dirty 状态保持不变，branch/HEAD 未改变。

- [V7 contract](../../outputs/20260911_v7/runs/lightgen_2d_transfer_aware_semantic_warmup_contract_v7_attempt_01/lightgen_2d_transfer_aware_semantic_warmup_contract.v7.json)，SHA-256=974d6dc560d2f4427cc75e5dc3d3404140d6e2a84439d800f6af10e299bc7766，READY/valid_completed。
- preflight attempt-03 [report](../../outputs/20260911_v7/runs/lightgen_2d_transfer_aware_semantic_warmup_preflight_v7_attempt_03/lightgen_2d_transfer_aware_semantic_warmup_v7_preflight_report.v1.json)，SHA-256=18ea57a4bd2baa5b9a975a5902f899e9bb8d377964ecbfde9dd01b3165aa332e；[lineage](../../outputs/20260911_v7/runs/lightgen_2d_transfer_aware_semantic_warmup_preflight_v7_attempt_03/lightgen_2d_transfer_aware_semantic_warmup_v7_preflight_lineage.v1.json)，SHA-256=fb1644c17fd1c6ae308ab5e9e5822e5196a165098f91ff34d8bc6cfe456da79f；scientific_300_step_executed=false。
- scientific input SHA-256=2f1ff0a200d6e7ef42859276ec2727aaf2ce36fe73d16f5fa6d4f8fa3d25ceb3；[report](../../outputs/20260911_v7/runs/lightgen_2d_transfer_aware_semantic_warmup_v7_scientific_attempt_01/lightgen_2d_transfer_aware_semantic_warmup_v7_scientific_report.v1.json)，SHA-256=ca06e5efe91c51f3b2ebff5125986e6599e607100f9c5932c4d504d55a2482fc；[lineage](../../outputs/20260911_v7/runs/lightgen_2d_transfer_aware_semantic_warmup_v7_scientific_attempt_01/lightgen_2d_transfer_aware_semantic_warmup_v7_scientific_lineage.v1.json)，SHA-256=4bac18f7b0bb588bb29cc75ab2087193ab2c44b26a7dc4e73f605085fe300139。

配置：exact V3 CONTROL step-300 encoder；exact V4 step-300 generator、V5 TRAIN-only prototypes、历史 TRAIN templates、OLS/T/W_OLS/Smooth 全冻结；仅 encoder.phase 可训练；Adam lr=1e-3、batch=10、300 updates、grid=2048；L_V7=L_transfer=mean(softplus(-Delta_template))；z_shift=z-mu_y+mu_t；t=(y+1) mod 10；image/prototype/Smooth loss 无梯度；DEV=5000，cyclic=1000/100，seed=4321。

运行设备：cuda:0，NVIDIA GeForce RTX 3060 Laptop GPU，compute capability=8.6，VRAM=6441926656 bytes，PyTorch=2.5.1，CUDA build=12.1。

| step | TRAIN L_transfer | TRAIN L_image | DEV L_image | Smooth macro | OLS margin mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.69342184 | 0.07510780 | 0.06706161 | 0.1036 | 0.00208249 |
| 100 | 0.69244408 | 0.06457431 | 0.06692999 | 0.1036 | 0.00207652 |
| 200 | 0.69080558 | 0.06446711 | 0.06679273 | 0.1036 | 0.00205580 |
| 300 | 0.68953269 | 0.06921547 | 0.06668990 | 0.1042 | 0.00202175 |

Natural Smooth Delta_R=+0.000600，95% CI=[-0.000600,+0.001800]，未达到 +0.05 Gate；DEV image Delta=-0.000371713；TRAIN transfer Delta=-0.00388916；OLS margin Delta=-6.07760e-5（secondary）。

cyclic transfer retained=true：template mean/median=0.00816400/0.00616277，95% CI=[0.00792898,0.00840002]；Smooth mean/median=2.53695471/2.49440956，95% CI=[2.43122264,2.64568621]；positive pair counts=10/10，broad=true，pair_dominated=false。十个 pair 的 template/Smooth means 依次为：0->1 0.02126103/2.63885640，1->2 0.01058828/3.32051782，2->3 0.00196776/0.22176126，3->4 0.00748423/4.37591817，4->5 0.00402526/2.81977525，5->6 0.00586092/2.10184672，6->7 0.00844589/3.64366655，7->8 0.00417849/1.49608160，8->9 0.00312327/1.42380861，9->0 0.01470491/3.32731475。

结果=VALID_COMPLETED/valid_completed，唯一分类=V7_TRANSFER_AWARE_SEMANTIC_WARMUP_NOT_SUPPORTED；Outer Gate V7_TRANSFER_AWARE_SEMANTIC_WARMUP_SUPPORTED 未满足。raw validity=true，但 natural Smooth primary=false；Stage-A firewall=R300=0.1042 < 0.937，STAGE_A_FIREWALL_FAILED。Stage B、generation、formal promotion 关闭。

preflight 与 scientific firewall 均确认 image-loss gradient=NO、prototype-loss gradient=NO、Smooth/Frozen gradient=NO、generator update=NO；negative-control transfer gradient finite/nonzero；无 retry、second arm、调参、target/magnitude sweep、scale/prior/generation、protected/test/sealed read 或 formal promotion。V5 latent-to-optical transfer-coupling mechanism 仍 unresolved。

## V5 latent-to-optical transfer-coupling diagnosis attempt-03（2026-09-11，VALID_COMPLETED；唯一科学分类）

按一次且仅一次明确授权 `DIAGNOSE_V5_LATENT_ALIGNMENT_TO_OPTICAL_TRANSFER_COUPLING_V1 attempt-03`，在
`research/lightgen-ols-native-semantic-generation-v1`、start/end HEAD=`a9cd1749adfa690e282fd52a6425564fc2cf102c` 上完成 forward-only/no-update scientific diagnosis。入口在 authority chain 中消费冻结 producer HEAD，但按授权不因 producer HEAD 与当前 consumer HEAD 不同而停止。

结果=`VALID_COMPLETED/valid_completed`，唯一科学分类=`V5_LATENT_TO_OPTICAL_TRANSFER_MIXED_OR_UNRESOLVED`；`V5_LATENT_TO_OPTICAL_TRANSFER_MECHANISM_RESOLVED` 仍 unresolved。attempt-01/02 的 INVALID report/lineage 保持 immutable，未重写、未重试。

- [`attempt-03 report`](../../outputs/20260911_v5/runs/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_v1_attempt_03/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_report.v1.json)，SHA-256=`b759edd433cf53961d8b1a4f9cf61a341fd5a75a56c0e2985969bcf1799f791e`；
- [`attempt-03 lineage`](../../outputs/20260911_v5/runs/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_v1_attempt_03/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_lineage.v1.json)，SHA-256=`71c11240c87b79478c047fd9d6688f0e2f5a789e3d76a2e616bccd72d22eb6bc`；
- [`sufficient observations`](../../outputs/20260911_v5/runs/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_v1_attempt_03/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_sufficient_observations.v1.pt)，SHA-256=`8985d1b48cb96e1f684ee263ebc050144802cf610c9806bed9cd21fc1c49b6ef`，schema reload=`PASS`；
- selection=`1000/100 per class`，`SELECTION_SEED=4321` reconstructed exact frozen selection-index SHA=`4acbae34a68778753d770f2cac955a8e109e438fb4df9f41e83b15df46890bb8`，producer omitted the seed field but identity matched；DEV scientific reads=`1000`；scientific forward=`600`（control encoder 100、candidate encoder 100、4 endpoints ×100）；Smooth endpoint evaluations=`400`。

全局几何摘要：`||delta_V||` mean/median=`8.618106e-6/8.632646e-6`（q05–q95=`5.653216e-6–1.161482e-5`）；`||delta_class||` mean/median=`6.110225e-5/6.064601e-5`；norm ratio mean/median=`0.152174/0.151641`；class-subspace `rho` mean/median=`0.228568/0.216745`；delta alignment mean/median=`0.089892/0.108981`。方向一致性为 `broad`，但不是正向 Smooth 证据。

| intervention | ΔS mean | median | paired 95% CI | positive fraction | primary flag |
|---|---:|---:|---:|---:|---|
| actual | -0.140839 | -0.086932 | [-0.154794, -0.125540] | 0.000 | false |
| matched-norm | -0.070122 | -0.071359 | [-0.084865, -0.056285] | 0.000 | false |
| full class-direction | -0.502025 | -0.485210 | [-0.620633, -0.398167] | 0.000 | false |

四个 raw endpoint 均 finite、shape=`[1000,1,200,200]`、structured、nonconstant。实际/匹配/完整方向相对 baseline 的 absolute RMS mean 分别为 `0.008265/0.008009/0.063604`，relative RMS mean 分别为 `0.062283/0.061172/0.540107`。Argmax recall 仅作描述：baseline/actual/match/full macro recall=`0.105/0.104/0.105/0.106`。

`ACTUAL_CONTINUOUS_SEMANTIC_GAIN=false`、`MATCHED_NORM_SEMANTIC_GAIN=false`、`FULL_CLASS_DIRECTION_SEMANTIC_GAIN=false`，所以不能支持 subthreshold、direction-misalignment 或 magnitude-limited 三个排他解释；按预注册规则归入 mixed/unresolved。没有 optimizer、backward、update、training、V5 retry、V6、Stage B、generation、protected/test/sealed access 或 formal promotion。

## V5 transfer-coupling production entry stabilization（2026-09-11，VALID_COMPLETED；非 scientific execution）

在 `research/lightgen-ols-native-semantic-generation-v1` 上完成一次 bounded engineering stabilization；起始 HEAD=`9a72e0bcf2de12ab9a0caa7151f273de76768dc7`，focused commit=`a9cd1749adfa690e282fd52a6425564fc2cf102c`，当前 HEAD=`a9cd1749adfa690e282fd52a6425564fc2cf102c`。入口文件当前 SHA-256=`ca7d0a987f6e2f01d6819511b4124980b91fa7fb02885e2a013ba31682f73c02`。

- [`stabilized V5 transfer-coupling entry`](../../scripts/research/diagnose_lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_v1.py) 已通过真实 authority schema dry-run：accepted preflight 的嵌套 `accepted_preflight.input.sha256`、valid V5 scientific attempt-01、注册 class-direction selection authority、V4 generator、V5 control/candidate checkpoint、Smooth v2.1、V2 runtime 均可解析；注册 selection producer 未声明 `selection_seed`，入口按已冻结 `SELECTION_SEED=4321` 与不可变 selection-index SHA 绑定，并记录 `seed_declared_by_producer=false`。
- production wrapper 的 deterministic synthetic acceptance 使用与未来真实路径相同的 resolved runtime/evaluator/核心统计，仅替换数据边界；12 次 wrapper forward、8 次 Smooth batch evaluation、sufficient/report/lineage reload 均通过。synthetic classification 仅为工程验收，scientific meaning=`none`。
- focused tests=`7 passed`；`py_compile`、Ruff、`git diff --check` 均通过。此次没有读取真实 TRAIN/DEV scientific rows，没有 scientific diagnosis forward，没有 optimizer/backward/update/training，也没有修改 attempt-01/02 report/lineage。

该工程稳定化在当时不产生 V5 transfer mechanism classification；随后 attempt-03 已在新的明确授权下完成并记录于上方。V5 mechanism 仍 unresolved，不开放 V6、Stage B、generation 或 formal promotion。

## V5 latent-to-optical transfer-coupling diagnosis attempt-02（2026-09-11，INVALID；不重试）

按新的明确授权 `DIAGNOSE_V5_LATENT_ALIGNMENT_TO_OPTICAL_TRANSFER_COUPLING_V1 attempt-02`，在
`research/lightgen-ols-native-semantic-generation-v1`、HEAD=`9a72e0bcf2de12ab9a0caa7151f273de76768dc7` 上执行一次；attempt-01 未修改。修复后的 authority binding 已通过静态检查，但入口因调用 accepted preflight 中不存在的 `_load_evaluators` 再次停止，结果=`INVALID/invalid_harness`，classification=`INVALID_V5_TRANSFER_COUPLING_UNEXPECTED_EXECUTION_ERROR`。

- [`attempt-02 diagnosis report`](../../outputs/20260911_v5/runs/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_v1_attempt_02/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_report.v1.json)，SHA-256=`0389608886eb7e8ac7c83e7560ba55447484548403d120c5209d4f73f55204d8`；
- [`attempt-02 diagnosis lineage`](../../outputs/20260911_v5/runs/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_v1_attempt_02/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_lineage.v1.json)，SHA-256=`85fdcdb668974e545d65b740d0cb062048e109956397da4888071265a323e37f`；
- attempt-02 执行时入口 SHA-256=`233e98e7036f71c36ed4c83c97dc2c4d5478a5dbea93304cce00fca529d361cd`，失败后修复后的当前入口 SHA-256=`8ef136a8e369e0ab81b0fc16fe53cd660e2d92950d6def1cd9f9b373cd8d1571`。

失败发生在 scientific forward 前：`diagnosis_executed=false`、`scientific_forward_count=0`、`diagnosis_started=false`；无 optimizer/backward/update/training，无 TRAIN/DEV scientific read，无 protected/official-test/sealed read，无 semantic classification。Outer Gate=`V5_LATENT_TO_OPTICAL_TRANSFER_MECHANISM_RESOLVED` remains unresolved。

失败后仅改为现有权威 `_load_smooth` 并保持 Smooth evaluator 冻结，未重写 attempt-01/02 历史报告或 lineage，未重跑。若继续必须重新授权新 attempt；本 attempt 不开放 V6、Stage B、generation 或 formal promotion。

## V5 latent-to-optical transfer-coupling diagnosis attempt-01（2026-09-11，INVALID；不重试）

按明确授权 `DIAGNOSE_V5_LATENT_ALIGNMENT_TO_OPTICAL_TRANSFER_COUPLING_V1`，在
`research/lightgen-ols-native-semantic-generation-v1`、HEAD=`9a72e0bcf2de12ab9a0caa7151f273de76768dc7` 上执行一次且仅一次入口诊断。synthetic exact-core preflight 已通过；真实 scientific 入口在 authority resolution 阶段因新 harness 错误停止，结果=`INVALID/invalid_harness`，classification=`INVALID_V5_TRANSFER_COUPLING_HARNESS`。

- [`diagnosis report`](../../outputs/20260911_v5/runs/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_v1_attempt_01/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_report.v1.json)，SHA-256=`49236400f58c8a15da6d87632a7601c7e936fd2fe1280be1be37713f03611eb0`；
- [`diagnosis lineage`](../../outputs/20260911_v5/runs/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_v1_attempt_01/lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_lineage.v1.json)，SHA-256=`74552a2a1fdb8d94cf30616dd51ebbee759d3c85bae7205cb0089046da43937d`；
- diagnosis entrypoint 在该 attempt 中的 SHA-256=`14a04ca9988e554629447404aa749ee0c68581c021055678c12686c41f0dc1cb`。

失败原因为入口错误读取 V5 scientific input 的 preflight authority 字段：实际为 `accepted_preflight.input.sha256`，入口却读取 `accepted_preflight.sha256`，所以在 scientific forward 前触发 `V5 input preflight binding changed`。`diagnosis_executed=false`、`scientific_forward_count=0`、`diagnosis_started=false`；无 optimizer/backward/update/training，无 TRAIN/DEV scientific read，无 protected/official-test/sealed read，无 semantic classification。Outer Gate=`V5_LATENT_TO_OPTICAL_TRANSFER_MECHANISM_RESOLVED` remains unresolved。

失败后仅修复了入口字段绑定，未改写本次 report/lineage，未重跑；若继续，需新的明确授权重新消费 one-shot diagnosis。该 INVALID attempt 不支持任何 V5 transfer mechanism classification，不开放 V6、Stage B、generation 或 formal promotion。

## V5 scientific execution 结果：direct complex-OLS semantic alignment（2026-09-11；历史 support snapshot）

按明确授权 `EXECUTE_LIGHTGEN_2D_COMPLEX_OLS_SEMANTIC_ALIGNMENT_V5`，在
`research/lightgen-ols-native-semantic-generation-v1`、HEAD=`9a72e0bcf2de12ab9a0caa7151f273de76768dc7` 上完成且仅完成一次 matched CUDA comparison：control/candidate 各 300 个真实更新，checkpoint 为 100/200/300；无 retry、调参、预算扩展或下游阶段。

结果=`VALID_COMPLETED/valid_completed`，classification=`V5_OLS_SEMANTIC_ALIGNMENT_NOT_SUPPORTED`；Outer Gate=`V5_OLS_SEMANTIC_ALIGNMENT_SUPPORTED` 未通过。

- [`scientific report`](../../outputs/20260911_v5/runs/lightgen_2d_complex_ols_semantic_alignment_v5_scientific_attempt_01/lightgen_2d_complex_ols_semantic_alignment_v5_scientific_report.v1.json)，SHA-256=`41aeb6e5ae5d1736521b164f2c8e9267bc3687801ab99b7142692112f8d85ce8`；
- [`scientific lineage`](../../outputs/20260911_v5/runs/lightgen_2d_complex_ols_semantic_alignment_v5_scientific_attempt_01/lightgen_2d_complex_ols_semantic_alignment_v5_scientific_lineage.v1.json)，SHA-256=`89e00a5f67dc082b0d2f37042a66a796cae71f14b1f676e8a27e94caaeed4c50`；
- [`scientific entrypoint`](../../scripts/research/execute_lightgen_2d_complex_ols_semantic_alignment_v5_scientific.py)，执行时 SHA-256=`b48d9f8ae9f128d9b2f7563e5892be8b82d7dc973ef9f56196a0ccf4571fae0f`；
- [`scientific input`](../../outputs/20260911_v5/runs/lightgen_2d_complex_ols_semantic_alignment_v5_scientific_attempt_01/lightgen_2d_complex_ols_semantic_alignment_v5_scientific_input.v1.json)，SHA-256=`91607325a08883479d49e69adac2d6a2c8ea6251ed86da5b45664aa0a7179b77`；paired bootstrap SHA-256=`2ded3c474df42c9188304669150977fcd2fa30b26ba493b40be3863ad6e46772`；sufficient observations SHA-256=`82104162a4058b789fc62a6028ba2d064bb6104c3d07732c05c2b42f6af138d9`。

固定 control=`L_image`、candidate=`L_image + 0.09573533218209243*L_proto`，仅训练 `encoder.phase`；generator、100-port complex OLS、`T=I_100`、shared `W_OLS` 与 Smooth evaluator 均冻结。TRAIN prototypes=`55,000`，`s_mu=0.00555040966719389`；DEV=`5,000`（每类 500）仅 evaluation；calibration/protected/test/sealed=`0`。运行环境为 RTX 3060 Laptop/compute capability `8.6`/PyTorch `2.5.1`/CUDA `12.1`/grid `2048`/batch `10`/Adam `lr=1e-3`。

| arm | step-100 `L_image` / Smooth / OLS margin | step-200 `L_image` / Smooth / OLS margin | step-300 `L_image` / Smooth / OLS margin |
| --- | --- | --- | --- |
| control | `0.0664125154` / `0.1040` / `0.0020675604` | `0.0658696759` / `0.1046` / `0.0020476440` | `0.0654365251` / `0.1044` / `0.0020292227` |
| candidate | `0.0668724473` / `0.1038` / `0.0021049360` | `0.0667107936` / `0.1042` / `0.0021195124` | `0.0665877977` / `0.1042` / `0.0021336675` |

step-300：Smooth Delta=`-0.0002000`，paired bootstrap `B=999`/seed=`271828`/95% CI=`[-0.0010,0.0004]`，最大 per-class degradation=`-0.004`；OLS margin Delta=`+0.0001044448`，CI=`[0.0001008631,0.0001078032]`；image-loss Delta=`+0.0011512727`（在 `+0.005` 容差内）；两端 raw-CMOS panel 均 finite/structurally valid/nonconstant 且 reload 通过。未通过项是 Smooth Delta `>=0.05` 与 Smooth CI 下界 `>0`，故整体 objective-correction classification 为 `NOT_SUPPORTED`。

candidate step-300 Smooth macro=`0.1042 < 0.937`，Stage-A firewall=`STAGE_A_FIREWALL_FAILED`。该结果是有效 bounded comparison failure，不是 harness INVALID；不授权 Stage B、formal scale、prior、sampled generation、generation acceptance 或 formal promotion，`generation_readiness=not_ready`。工作区原有未提交变更保持，未自动 commit。

## V5 状态增量：direct complex-OLS semantic alignment definition 与工程预检（科学执行前 authority）

V5 semantic authority=`DEFINE_LIGHTGEN_2D_COMPLEX_OLS_SEMANTIC_ALIGNMENT_CONTRACT_V5` 已完成 definition-only 合同，结果为
`READY/valid_completed`，classification=`DEFINE_LIGHTGEN_2D_COMPLEX_OLS_SEMANTIC_ALIGNMENT_CONTRACT_V5_READY`；未来 Outer Gate 为
`V5_OLS_SEMANTIC_ALIGNMENT_SUPPORTED`。合同证据为
[`V5 contract`](../../outputs/20260911_v5/runs/lightgen_2d_complex_ols_semantic_alignment_contract_v5_attempt_01/lightgen_2d_complex_ols_semantic_alignment_contract.v5.json)，
SHA-256=`e2d57fa96e0a020abf23cbd889188d74d4893a6e9c48ac7a1d24d7156cf21b61`。

合同冻结 V3 CONTROL step-300 encoder、有效 V4 step-300 fixed-OLS generator、complex `C^100` OLS、modal projection/re-emission、
`T=I_100`、shared `W_OLS`、现有 gauge/distance 与 ten TRAIN prototypes；两臂只训练 `encoder.phase`，control=`L_image`，candidate=`L_image + lambda_proto*L_proto`。
TRAIN 原型使用 `55,000` 行，`s_mu=0.00555040966719389`；首个 TRAIN batch 一次性校准
`g_image=0.0005046187151008359`、`g_proto=0.005270976801389226`、`lambda_proto=0.09573533218209243`，不调参、不重算。

最终 bounded engineering preflight 为 attempt-05，结果=`valid_completed/AUTHORIZABLE`，classification=
`LIGHTGEN_2D_COMPLEX_OLS_SEMANTIC_ALIGNMENT_V5_AUTHORIZABLE`：

- [`input`](../../outputs/20260911_v5/runs/lightgen_2d_complex_ols_semantic_alignment_preflight_v5_attempt_05/lightgen_2d_complex_ols_semantic_alignment_preflight_input.v1.json)，SHA-256=`9c734a76b5ceb03fe6405faadfaf7d2fa27eefe21d5a077ab3f0e46d926569f8`；
- [`report`](../../outputs/20260911_v5/runs/lightgen_2d_complex_ols_semantic_alignment_preflight_v5_attempt_05/lightgen_2d_complex_ols_semantic_alignment_preflight_report.v1.json)，SHA-256=`7fab2789b36491597462ce092bc98dc5784cc269245a2cab3b07a4878543c551`；
- [`lineage`](../../outputs/20260911_v5/runs/lightgen_2d_complex_ols_semantic_alignment_preflight_v5_attempt_05/lightgen_2d_complex_ols_semantic_alignment_preflight_lineage.v1.json)，SHA-256=`e5d72fce35784ea3c25fc98a5c4a7ad20538594f491f9065123a94f5892387dc`。

该预检已通过 RTX 3060 Laptop/8.6/PyTorch 2.5.1/CUDA 12.1 的 production 2048-grid no-data smoke；control/candidate 各真实 CUDA 三步，
encoder-only、generator/OLS freeze、candidate prototype gradient、raw CMOS 与 checkpoint/resume identity 均通过，resume 的 next batch、optimizer、
runtime、raw、trace 差异为 `0`。`lambda_proto` 已按固定梯度规则得到，非科学 synthetic exact-core 也覆盖了 prototypes/scale/loss、两臂、firewall、
checkpoint/evaluator/margin/bootstrap/statistics/sufficient-observation reload/schema。TRAIN=`60,000` loaded / `55,000` used / each arm `30` update rows；
DEV scientific values=`0`，protected/test/sealed=`0`；CUDA DLL 仅在当前进程使用 `sys.prefix\\bin` 修复。

`AUTHORIZABLE` 只表示工程入口可执行，不是 `V5_OLS_SEMANTIC_ALIGNMENT_SUPPORTED`、Stage-A eligibility、Stage-B authorization、generation readiness 或 formal promotion；上方已记录随后一次 scientific execution 的有效结果。
`stage_b_authorized=false`、`generation_readiness=not_ready`、`formal_promotion=false` 保持。attempt-01 至 attempt-04 的 harness INVALID 结果均保留为 immutable 历史，不覆盖 attempt-05，且不构成科学结论。

## 本次目标增量

用户明确要求：输入与 OLS 对齐陈一同等人的 LightGen，衍射层架构可以不同；尽快得到语义生成原型。
唯一活动研究主线及路线地位见 [项目记忆](../PROJECT_MEMORY.md#唯一活动研究主线lightgen-输入与-ols-对齐的语义生成原型)，
已完成科学结果统一查 [当前状态](../CURRENT_STATUS.md)。不把正式 `label + 8D` 基线或历史 reconstruction
closeout 当成当前研究目标；也不把 N3 分布支持当成 LightGen 输入/OLS 对齐完成。当前交接增量是：基于 V2
Stage-A attempt-02 与 warmup diagnosis attempt-02，定义并完成一项固定架构的 V3 semantic objective-correction 工程 preflight；随后按独立授权完成一次 V3 scientific run 与一次 no-update failure attribution，V3 结果为有效科学失败，归因为 provider-specific alignment；在此基础上定义 V4 fixed-OLS generator-recoverability contract，完成修复后 accepted engineering preflight，并按新的独立授权完成 V4 scientific attempt-02，结果为有效科学失败 `FIXED_OLS_GENERATOR_RECOVERABILITY_NOT_SUPPORTED`。attempt-01 的 `INVALID_HARNESS` 保留为历史证据，不重复 V2/V3。
随后按独立授权启动一次 `DIAGNOSE_COMPLEX_OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_V1`；attempt-01 在任何 TRAIN/DEV 科学读取或 forward 前因 stale loader harness 错误 INVALID，修复后 attempt-02 又在 CUDA device preflight 因 NVRTC DLL 不可用 INVALID；readiness 核查通过后，attempt-03 进入 scientific forward，但在 raw-response 汇总时因缺失 `_raw_summary` helper 以 `INVALID/invalid_harness` 停止。三次 one-shot 均已消费且不重试。随后在稳定化 runner 上按独立 replication 授权完成一次冻结语义 replication，结果为 `VALID_COMPLETED / OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_SUPPORTED`。后续工程核查确认实际 2048 LightGen optical path 同样触发该环境 DLL 搜索问题；使用当前 `sys.prefix/bin` 已有 DLL 的进程级 PATH 修复后，真实合成 no-data readiness smoke 与 replication 均通过，Stage B/generation/promotion 仍关闭。

## 当前状态增量：V2 Stage-A attempt-02 与 diagnosis（2026-09-11）

V2 Stage-A attempt-02 已按 corrected authority 完成 300 个真实更新，结果为
`VALID_COMPLETED/valid_failed`，classification=`LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_V2_SEMANTIC_WARMUP_NOT_ESTABLISHED`。
报告、lineage 与 diagnosis 为：

- [`V2 training report`](../../outputs/20260911_v2/runs/lightgen_2d_semantic_complex_ols_training_v2_attempt_02/lightgen_2d_semantic_complex_ols_training_v2_report.v1.json)，SHA-256=`de55a0f307675ded10aca5726f9ac09474633b2c9a5f8bfb1890075de53a530d`；lineage SHA-256=`899aa78daff4bbe2f70cf72e01080e323e24d791f4937ca35fefc8c9b30394dc`；
- [`warmup diagnosis report`](../../outputs/20260911_v2/runs/lightgen_2d_semantic_ols_warmup_failure_diagnosis_v1_attempt_02/lightgen_2d_semantic_ols_warmup_failure_diagnosis_report.v1.json)，SHA-256=`f05eb4e6dfd7c8c3ce1865ff8097fdef2dfa649b3e3dab3352f5282fa544f3da`；diagnosis lineage SHA-256=`ef72fcaf191fc147d34fc460fb0d0f147614b2d696d677feae7d5d501f40b763`。

DEV macro recall=`0.170 → 0.160 → 0.136`，DEV image loss=`0.0711633662 → 0.0689440685 → 0.0673580589`，checkpoint 为 100/200/300；Stage B、
generation、protected/test/sealed read 均为 0。诊断 primary 为
`SEMANTIC_WARMUP_FAILURE_IMAGE_OBJECTIVE_MISMATCH_SUPPORTED`；梯度有限且非零，故只支持 objective mismatch，不支持容量、OLS 或 Stage-B 结论。
此前交接条目中“下一步执行 V2 training”的文字是历史状态，已被本段的 V3 objective-correction 定义取代；历史事件本身不覆盖。

## 当前 V3 definition：semantic objective correction（2026-09-11）

V3 semantic authority ID=`DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_OBJECTIVE_CORRECTION_CONTRACT_V3`，结果为
`READY/valid_completed`，classification=`DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_OBJECTIVE_CORRECTION_CONTRACT_V3_READY`。
合同为
[`lightgen_2d_semantic_complex_ols_objective_correction_contract.v3.json`](../../outputs/20260911_v2/runs/lightgen_2d_semantic_complex_ols_objective_correction_contract_v3_attempt_01/lightgen_2d_semantic_complex_ols_objective_correction_contract.v3.json)，
SHA-256=`e537fbf4b856519a71e48c16a48c149fadd327e2ad12db84f64c953413fb53f6`；Outer Gate=
`V3_SEMANTIC_OBJECTIVE_CORRECTION_SUPPORTED`。

control 固定为纯 `L_image`，candidate 固定为 `L_image + lambda_feature*L_feature`；provider 是冻结的 128D
`FrozenMNISTEvaluator v1` embedding，独立 `SmoothMNISTEvaluator v2.1` 只作 metric corroboration。λ 在共享初始状态的 TRAIN step-1 batch 上按
`min(10, max(0.1, g_image/g_feature))` 预先计算一次；两臂共享零 phase 初始化、同一 batch schedule、Adam、300-step Stage-A budget 和 100/200/300 checkpoints。
主要支持条件为 step-300 Smooth macro recall 提升至少 `0.05`、999 次 paired stratified bootstrap 95% CI 下界大于 0、无 per-class 下降超过 `0.10`、
image loss 容差 `+0.005` 内，且 raw-CMOS endpoint panel 有限/结构有效/非恒定。未满足则返回
`V3_SEMANTIC_OBJECTIVE_CORRECTION_NOT_SUPPORTED`；这不等价于 Stage-A eligibility。

Stage-A→B firewall 继续固定为 Frozen evaluator macro recall `>=0.937`，不降低、不补偿；本定义不授权 training、Stage B、generation、scale extraction 或 formal promotion。
后续仅在独立明确授权后执行 V3 future entrypoint 的 bounded engineering preflight，先验证三步真实 CUDA 更新、梯度/冻结参数 firewall、matched pair、disabled-term 与
detached-provider negative controls、以及 checkpoint/resume identity。

## 当前 V3 bounded engineering preflight 结果（2026-09-11）

按 V3 READY contract 完成最终 attempt-09 的工程 preflight，结果为
`valid_completed/AUTHORIZABLE`，classification=`LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_OBJECTIVE_CORRECTION_V3_AUTHORIZABLE`。
`AUTHORIZABLE` 仅表示冻结的未来 paired Stage-A 入口在当前环境中技术上可执行；不表示
`V3_SEMANTIC_OBJECTIVE_CORRECTION_SUPPORTED`，不授权 scientific training、Stage-A eligibility、Stage B、generation 或 promotion。

证据为：

- [`preflight input`](../../outputs/20260911_v3/runs/lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_attempt_09/lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_input.v1.json)，SHA-256=`631eb85792062f00e2e59344d091a6aac203225e05c56e1a4c27150859772d36`；
- [`preflight report`](../../outputs/20260911_v3/runs/lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_attempt_09/lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_report.v1.json)，SHA-256=`265e9b80582161872a207041cd5edc2279b6f72c5729aa6f5406f9646e354468`；
- [`preflight lineage`](../../outputs/20260911_v3/runs/lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_attempt_09/lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_lineage.v1.json)，SHA-256=`f600bf924c3494f0b348fccde19aa6d49d5b33923cff622d5a5722394cb08e31`；
- [`preflight entrypoint`](../../scripts/research/execute_lightgen_2d_semantic_complex_ols_objective_correction_v3.py)。
- [`control checkpoint`](../../outputs/20260911_v3/runs/lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_attempt_09/lightgen_2d_semantic_complex_ols_objective_correction_v3_control_checkpoint.v1.pt)，SHA-256=`ac4d1e2521d4f265472aa4ba36673e19d42e16c8972b5bde24c24b5273a79c3a`；
- [`candidate checkpoint`](../../outputs/20260911_v3/runs/lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_attempt_09/lightgen_2d_semantic_complex_ols_objective_correction_v3_candidate_checkpoint.v1.pt)，SHA-256=`b16011a5b443f16b31399c0e0b4d3a220402725b51e49eb1df52b449bc9b8c7f`。

目标设备为 RTX 3060 Laptop GPU、compute capability=`8.6`、PyTorch=`2.5.1`、CUDA build=`12.1`、VRAM=`6,441,926,656` bytes；
grid=`2048`、batch=`10`、Adam lr=`1e-3`。control 为 `L_image`，candidate 为
`L_image + lambda_feature*L_feature`；FrozenMNISTEvaluator v1 的 128D normalized embedding 作为冻结 provider，SmoothMNISTEvaluator v2.1
仅保留为未来独立 metric corroboration。λ 在共享初始状态/首个 TRAIN batch、任何 optimizer update 前一次性计算：
`g_image=0.0012350937723848974`、`g_feature=0.0631487965185555`、ratio=`0.01955846889373387`，固定得到
`lambda_feature=0.1`。

control 与 candidate 各完成 3 次真实 CUDA 更新；两组 trainable phase 均有限且非零更新，candidate semantic input gradient 三步均有限非零，
modal projection → `T=I_100` → `W_OLS` → generator 路径通过。disabled-term control 通过；detached-provider negative control 的 feature forward
仍有限但 semantic gradient 被移除，按合同标记为应拒绝的 `invalid_harness`。provider、independent evaluator 和 runtime non-phase state 未更新。
step-2 checkpoint 的 save/reload/resume 中 next batch、sampler、RNG、lambda、provider identity 通过；candidate 的 trace 最大差异约
`5.62e-10`、trainable parameter 最大差异约 `1.65e-8`，在预检确定性容差内通过。

本次 preflight 没有执行 300-step scientific Stage-A、科学 control/candidate comparison、DEV semantic selection、Stage B、scale/prior/generation、
protected/test/sealed read 或 promotion；`training_authorized=false`、`stage_b_authorized=false`、`generation_readiness=not_ready`、
`formal_promotion=false` 保持。attempt-01/02 为 harness 适配修复，attempt-03 为 CUDA DLL 环境阻断，attempt-04 至 attempt-08 为恢复比较器修复；
均为 immutable 历史 artifacts，不覆盖 attempt-09 的最终 preflight 结果。V3 scientific 已在其后按独立授权一次性完成，结果见下一节。

## 当前 V3 Stage-A scientific objective comparison 结果（2026-09-11）

执行 scope=`EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_OBJECTIVE_CORRECTION_V3` 已从冻结零 phase 初始状态完成一次 control/candidate 各 300 个真实 CUDA updates；
没有重试、参数重选、Stage B、scale/prior/generation 或 protected/test/sealed read。结果为 `VALID_COMPLETED`，classification=
`V3_SEMANTIC_OBJECTIVE_CORRECTION_NOT_SUPPORTED`。

- [`scientific report`](../../outputs/20260911_v3/runs/lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_attempt_01/lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_report.v1.json)，SHA-256=`8d7930449a1872e685ad39dafa499d89fdcc1d6bb5a9317e7ba06c554f54f31a`；
- [`scientific lineage`](../../outputs/20260911_v3/runs/lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_attempt_01/lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_lineage.v1.json)，SHA-256=`00127edfad105e077342a8eb270447ea577cd4c48b816fb0435c0e6edbfc237d`；
- [`scientific entrypoint`](../../scripts/research/execute_lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific.py)，SHA-256=`ca9eb8db3c7360f31da8d08ed6d0f61fa3bc61e8f0ce3de0f5a952803ff44ae4`。

step-300 primary endpoint：Smooth control/candidate=`0.103/0.107`，`Delta_smooth=+0.004`；paired bootstrap `B=999`、seed=`271828` 的 95% CI=`[0, 0.009]`；
image loss control/candidate=`0.0673580589/0.0726336428`，差值=`+0.0052755839`；Smooth per-class 最大下降未超过 `0.10`，raw-CMOS panels 均有效且非恒定。
Frozen evaluator macro recall control/candidate=`0.136/0.283`，但 candidate 仍低于不可补偿的 `0.937` firewall，故 Stage-A firewall=`FAILED`。

因此本次只证明该冻结 single-term feature objective 在本预算/条件下未达到预注册 objective-correction 支持标准；不支持 Stage-A eligibility 或 Stage-B。
6 个 checkpoint、6 个固定 `[1000,1,200,200]` raw-CMOS panels 与 bootstrap artifact 已回读并通过 hash/shape/finite 核验。
`training_authorized=false`、`stage_b_authorized=false`、`generation_readiness=not_ready`、`formal_promotion=false` 保持；不得自动重试或由失败结果进入下一阶段，
任何新 objective/weight/architecture 实验必须重新定义合同并取得独立授权。

## 当前 V3 no-update failure attribution 结果（2026-09-11）

在上述有效科学失败之后，按 `RESOLVE_V3_OBJECTIVE_CORRECTION_FAILURE_ATTRIBUTION_V1` 对同一批冻结的 step-300
control/candidate checkpoint 完成一次且仅一次 forward-only 诊断。Outer Gate=`V3_OBJECTIVE_FAILURE_ATTRIBUTION_RESOLVED`，结果为
`VALID_COMPLETED`，classification=`PROVIDER_SPECIFIC_ALIGNMENT_SUPPORTED`。

- [`attribution input`](../../outputs/20260911_v3/runs/lightgen_2d_semantic_complex_ols_objective_correction_failure_attribution_v1_attempt_05/lightgen_2d_semantic_complex_ols_objective_correction_failure_attribution_input.v1.json)，SHA-256=`0c4945ca91641e89de44af3c131106d5b8fa55d42dbe842244b57e1b40ced794`；
- [`attribution report`](../../outputs/20260911_v3/runs/lightgen_2d_semantic_complex_ols_objective_correction_failure_attribution_v1_attempt_05/lightgen_2d_semantic_complex_ols_objective_correction_failure_attribution_report.v1.json)，SHA-256=`b3175da21e50448c5e6eeb0c2c36042f6b620c2510865c5059de26eb43277393`；
- [`attribution lineage`](../../outputs/20260911_v3/runs/lightgen_2d_semantic_complex_ols_objective_correction_failure_attribution_v1_attempt_05/lightgen_2d_semantic_complex_ols_objective_correction_failure_attribution_lineage.v1.json)，SHA-256=`21c095047ce119f69795cb543acb5ac5fac5199380c383024ec8e6638cda9be5`。

Matched fixed DEV：Frozen macro=`0.136→0.283`，Delta=`+0.147`，CI=`[0.119,0.17005]`；独立 Smooth macro=`0.103→0.107`，
Delta=`+0.004`，CI=`[0,0.009]`；Frozen 128D feature distance Delta=`-0.1388709251`，CI=`[-0.1496474,-0.1278142]`。
2×2 cross-swap 的 Smooth macro 为 `0.103/0.107/0.102/0.107`（control+control/candidate+candidate/candidate+control/control+candidate），
四条 raw-CMOS 路径均 finite/structurally valid/nonconstant。gauge `S(z)∈R^200` effective rank=`15.5868/15.8340`，between/within ratio=`0.959968/0.961896`；
固定 TRAIN centroid→DEV nearest-centroid macro=`0.761→0.763`，Delta=`+0.002`，CI=`[-0.005,0.010]`。因此 provider 的改善没有转移到独立 evaluator，
且 centroid/geometry 未建立 material latent semantic gain；这只是 failure attribution，不是语义生成、Stage-A eligibility 或 Stage-B 证据。

诊断无 optimizer、backward 或 optical update；TRAIN centroid 使用每类 source-index-sorted 前 100 行，DEV 使用原固定 1000 行，calibration/protected/test/sealed read 均为 0。
Frozen Stage-A firewall 仍为 `0.937`，candidate=`0.283`，`stage_b_authorized=false`；不得自动重训、重试、scale/prior/generation 或 formal promotion。

## 当前 V4 fixed-OLS generator-recoverability contract 与工程预检（2026-09-11）

V3 scientific failure 与 attribution 已解析后，定义了一个且仅一个 fixed-OLS generator-recoverability experiment。它冻结 V3
CONTROL step-300 encoder 与真实复数 `z∈C^100`，恢复 corrected-V2 runtime 的原始 zero-phase generator，仅训练
`generator.phase`，目标固定为 `L_generator=L_image`。100-port complex OLS、mandatory modal projection/re-emission、
`T=I_100`、shared `W_OLS`、2048 grid、TRAIN order、Adam `lr=1e-3` 与不可降低的 `0.937` Stage-A firewall 均不变；semantic
provider 不提供训练梯度，仅作 frozen/evaluation-only smoke。

V4 authority ID=`DEFINE_LIGHTGEN_2D_FIXED_OLS_GENERATOR_RECOVERABILITY_CONTRACT_V4`，结果为
`READY/valid_completed`，classification=`DEFINE_LIGHTGEN_2D_FIXED_OLS_GENERATOR_RECOVERABILITY_CONTRACT_V4_READY`，
Outer Gate=`FIXED_OLS_GENERATOR_RECOVERABILITY_SUPPORTED`。合同为
[`lightgen_2d_fixed_ols_generator_recoverability_contract.v4.json`](../../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_generator_recoverability_contract_v4_attempt_01/lightgen_2d_fixed_ols_generator_recoverability_contract.v4.json)，
SHA-256=`d46740ded4db58d7d9f0caa8acaf40942d03dc3407d9611d24efcaa12d3036b8`。

合同绑定 V3 CONTROL checkpoint SHA-256=`a42a43fdf99cf12d9abd1917a6d025fc2fcb9ecee39f0c05429dd8d7a8870079`、encoder.phase
SHA-256=`61648ce609ce4e62f890112faf3f3f91ffa5c018e630ec2d451a3d19c11ed9ca`；fresh original zero-phase runtime state
SHA-256=`0f90086cf26a179905a27a5a519f7599de256e4f4225f7798f24aeed086eb813`，generator.phase zero tensor SHA-256=
`68c468f955fb6606cd38278ff4ee4fb9f4a0c6c14337f59c72fc9be2eabb32d9`。V4 trainable surface 只有 `generator.phase` 的
`20,000` 参数；encoder `20,000` 参数、OLS frozen state `2,010,000` scalars 均冻结。该 contract 是 definition-only，
`training_authorized=false`，不打开 scientific run、Stage B、generation 或 promotion。

最终 accepted engineering preflight 为 attempt-02，结果为
`valid_completed/AUTHORIZABLE`，classification=`FIXED_OLS_GENERATOR_RECOVERABILITY_PREFLIGHT_AUTHORIZABLE`：

- [`input`](../../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_generator_recoverability_preflight_v4_attempt_02/lightgen_2d_fixed_ols_generator_recoverability_preflight_input.v1.json)，SHA-256=`ef3388a575c1eb648baef1c60e8932d6d0d51805af09a784d68dc14e7739e626`；
- [`report`](../../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_generator_recoverability_preflight_v4_attempt_02/lightgen_2d_fixed_ols_generator_recoverability_preflight_report.v1.json)，SHA-256=`c2b0d852e1178915523ceee21172b10b8d462b6d0096ecf87c8443c410c7b7fb`；
- [`lineage`](../../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_generator_recoverability_preflight_v4_attempt_02/lightgen_2d_fixed_ols_generator_recoverability_preflight_lineage.v1.json)，SHA-256=`139fd231c73cb5d8f90e74508e6d8d13645b7a6c9c30d839ad7311cbba642da8`；
- [`step-2 checkpoint`](../../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_generator_recoverability_preflight_v4_attempt_02/lightgen_2d_fixed_ols_generator_recoverability_preflight_step_2_checkpoint.v1.pt)，SHA-256=`00bd0af0f2c8bbcded39215bb6cf53bfc0fcd64efb952d4b693557ca0308dcec`；
- [`preflight entrypoint`](../../scripts/research/execute_lightgen_2d_fixed_ols_generator_recoverability_v4.py)，SHA-256=`8f99e098e121ab3dc8b667ddcd9c1dfbf57d1687fbfd8b87e166cf3bbb2d5b73`。

3-step continuous generator-only CUDA trace 通过：generator gradient norm=`0.0009246/0.0006487/0.0008156`，encoder gradient=`None`，
generator update norm=`0.13147/0.10217/0.08960`，encoder/OLS 每步 unchanged，modal projection/T=I_100/W_OLS provenance 有效。
canonical frozen-encoder z 与 cached-z replay 最大差异=`0`，raw replay 最大差异=`0`；step-2 save/reload/resume 的 step-3 loss、gradient、
update、raw、runtime state 与 optimizer-state 最大差异均=`0`。Frozen/Smooth evaluator 只作 TRAIN smoke no-grad forward，状态未变且无梯度；
TRAIN consumed 30 rows，DEV values、protected/test/sealed 均未读。

attempt-01 因 complex state comparator 的校验实现发出不完整复数比较警告，不作为最终证据；修复后 attempt-02 才是 accepted authority。
V4 preflight 仅证明未来 bounded scientific entrypoint 的工程边界可执行，不是 `FIXED_OLS_GENERATOR_RECOVERABILITY_SUPPORTED` 科学结果，
不证明 final LightGen/BOGT/generation readiness，不授权 Stage B、scale/prior/generation 或 formal promotion。下一步若要执行科学 300-step
实验，必须基于本 V4 contract 取得独立明确授权；不得自动启动。

## 当前 V4 scientific attempt-02 结果（2026-09-11）

用户随后独立明确授权 `EXECUTE_LIGHTGEN_2D_FIXED_OLS_GENERATOR_RECOVERABILITY_V4`，授权仅限新的
scientific attempt-02。attempt-01 保留为不可覆盖的 `BLOCKED/INVALID_HARNESS`，scientific steps=`0/300`；其 report SHA-256=
`9887f93bd942c9e4cf21017aa5e443864bc4bcc8862390bf41023f13184f62b2`，lineage SHA-256=`70780327719c6314c7515db3386d23d806294b5be5021baca6f81225b875ec81`。
修复后的 entrypoint SHA-256=`a2a54556348deddc21a5ae0caf5c1343c9f6736f73d211aa33a1958ffb33375b`；bounded reopening check 的
contract、preflight、V3 CONTROL、zero-phase 与 RNG identity 全部通过。

attempt-02 精确完成 300 个真实 CUDA generator-only updates，结果为
`VALID_COMPLETED`，classification=`FIXED_OLS_GENERATOR_RECOVERABILITY_NOT_SUPPORTED`。产物为：

- [`scientific input`](../../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_generator_recoverability_scientific_attempt_02/lightgen_2d_fixed_ols_generator_recoverability_scientific_input.v1.json)，SHA-256=`8acd28ef99983e97d37f1d97a7ca39bf50d159ad358da520dbdfdae04a4ba2f9`；
- [`scientific report`](../../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_generator_recoverability_scientific_attempt_02/lightgen_2d_fixed_ols_generator_recoverability_scientific_report.v1.json)，SHA-256=`fdfeb8d800a2d806f3ecc56503ef8d66cc68f06de6cd9e47aa4eab7c93ca2784`；
- [`scientific lineage`](../../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_generator_recoverability_scientific_attempt_02/lightgen_2d_fixed_ols_generator_recoverability_scientific_lineage.v1.json)，SHA-256=`6089bc41797580aa95149b0478a0e25190a450e00ec4a47c886b756f0187f9e6`。

运行条件为 RTX 3060 Laptop GPU、compute capability=`8.6`、PyTorch=`2.5.1`、CUDA build=`12.1`、grid=`2048`、
batch=`10`、Adam `lr=1e-3`；仅 `generator.phase` 的 `20,000` 参数可训练，encoder、OLS/runtime、Frozen/Smooth evaluator
均保持冻结/unchanged，semantic provider gradient=`false`。初始 RNG digest 与冻结 authority 均为
`6241d4343843975f8306769ff6ff05c0cb6af5b86956b545892592cf0e65ea30`，CPU/CUDA state identity 与
`cuda_available=true` 均通过。

固定 DEV trajectory：step 100/200/300 的 TRAIN image loss 为
`0.0656021386/0.0654420778/0.0698628575`，DEV image loss 为
`0.0693966536/0.0682117225/0.0673444915`，Smooth macro 为
`0.105/0.104/0.103`。step-300 相对 V3 CONTROL 的 Smooth=`0.103→0.103`，delta=`0`，paired bootstrap
95% CI=`[0,0]`，per-class delta 全部为 `0`，最大 degradation=`0`；DEV image-loss delta=`-0.0000135674`；raw-CMOS
finite/shape/structured/nonconstant 全部通过。

Gate 判定为：Smooth improvement=`FAIL`（`0 < 0.05`）；bootstrap lower=`FAIL`（`0` 不大于 `0`）；per-class preservation=`PASS`；
image-loss tolerance=`PASS`；raw-CMOS validity=`PASS`。因此 fixed-OLS recoverability Outer Gate 未通过。独立 Stage-A
firewall 为 Smooth=`0.103 < 0.937`，`STAGE_A_FIREWALL_FAILED`；Stage B 仍未授权。

数据 firewall：TRAIN optimization=`55,000`，DEV evaluation selection=`1,000`（DEV role=`5,000`），calibration/protected/
test/sealed=`0`。没有执行 Stage B、formal scale extraction、prior shaping、sampled-OLS generation、generation acceptance 或
formal promotion；`training_authorized=false`、`stage_b_authorized=false`、`generation_readiness=not_ready`、
`formal_promotion=false`。合同规定的 attempt-02 已完成且不得 retry；任何后续 objective、architecture 或 generation 动作必须新定义合同并重新取得独立授权。

## 当前 research provenance 治理：HASH_ON_TRUST_BOUNDARY_V2（2026-09-11）

本轮治理增量不执行 V4 conditionality diagnosis。普通 research handoff 现在默认使用 semantic authority ID、producer
task/attempt 与 Git branch/HEAD；完整 child SHA-256 仍保留在 machine manifest/registry、formal evidence、checkpoint
transfer 与 write-once artifact boundary，只有真实 consumer 要求且身份域相同时才核验。

`file_byte_sha256`、`state_dict_tree_sha256`、`tensor_content_sha256`、`semantic_state_identity` 与 `manifest_sha256`
互不等价。未被 consumer 要求的 hash 关系记录为 `NOT_REQUIRED` / `NOT_APPLICABLE`；跨域 hash 不一致不再自动变成
`INVALID_HARNESS` 或 `HARD_BLOCKER`，同域且明确要求的 authoritative mismatch 仍 fail-closed。

V4 conditionality diagnosis attempt-01 的 immutable 结果保持 `INVALID`：`diagnosis_executed=false`、scientific forward=`0`、
TRAIN/DEV diagnostic reads=`0`。入口的错误 Smooth file-SHA/state-tree-SHA equality 已修正，并由 focused regression tests 覆盖；
没有 retry、没有科学结论，`V4_GENERATOR_CONDITIONALITY_RESOLVED` 仍 unresolved。

Phase A 已在新治理 HEAD `1338ef5bad1936edd1f98fb3577933ed9bbd524a` 形成唯一 focused commit。随后按授权对
`DIAGNOSE_V4_FIXED_OLS_GENERATOR_CONDITIONALITY_V1` 执行 attempt-02 一次；它在 authority resolution 阶段发现不可变 V4
contract 仍要求旧 HEAD `383197a930dd48f98294120f8cf867444f994f85`，故结果为 `INVALID`，没有写入 diagnosis input、
scientific forward=`0`、TRAIN/DEV diagnostic reads=`0`，没有 semantic classification。report SHA-256=
`db2d6cfbe98cf7a0b732d6c9ae09d69931e63e77c7250149cdd3ce44d479d29c`，lineage SHA-256=
`8f23bca691b459d1ee24e78dd3e6131e3bb3cde1139a536ab52fcb0373ce09ba`。该不可变 attempt-02 不重试；
`V4_GENERATOR_CONDITIONALITY_RESOLVED` 仍 unresolved，未授权 training、Stage B、generation 或 formal promotion。

随后完成的 provenance consumer 修复已在 HEAD `1338ef5bad1936edd1f98fb3577933ed9bbd524a` 上验证：历史 V4
contract 的 `git.head` 作为 producer/contract provenance 保留，consumer branch/HEAD 单独记录；没有显式
`consumer_head_exact_required=true`（或等价 exact-source-lock）时，producer/consumer HEAD 差异允许。显式 exact lock
仍 fail-closed；V4 contract 及 attempt-01/02 immutable artifacts 未修改。本轮没有执行 V4 diagnosis 或任何 scientific
forward/data read，仅完成 focused tests、compile、Ruff 和 diff check；`V4_GENERATOR_CONDITIONALITY_RESOLVED` 仍 unresolved。

## 当前 V4 generator conditionality diagnosis attempt-03（2026-09-11）

按独立授权执行 `DIAGNOSE_V4_FIXED_OLS_GENERATOR_CONDITIONALITY_V1` attempt-03 一次且仅一次；consumer HEAD=
`054026ad3b44558b108888178ac2fb58aebb2f52`，历史 V4 contract producer HEAD=`383197a930dd48f98294120f8cf867444f994f85`，
`consumer_head_exact_required=false`，两者分离记录。结果为 `VALID_COMPLETED`，primary classification=
`V4_GENERATOR_CONDITIONALITY_MIXED_OR_UNRESOLVED`。

Matched MSE mean/median=`0.0673444926/0.0652310392`；within pairing `Delta` mean/median=`0.0056112921/0.0023668033`，
CI=`[0.0048836686,0.0063864972]`，ranking=`0.632`，raw RMS mean=`0.0961877042`；cross pairing `Delta` mean/median=
`0.0093417037/0.0059634257`，CI=`[0.0083926659,0.0102811366]`，ranking=`0.720`，raw RMS mean=`0.1161952598`；
`Delta_class` mean=`0.0037304116`，CI=`[0.0030554547,0.0044253468]`。预注册 checks 为 collapse=false、
class_conditioning=false、instance_conditioning=false，因此不能唯一判定 class-level 或 instance-level conditioning。
Smooth 仅 descriptive：matched/within/cross target-label macro=`0.103/0.103/0.098`，independent semantic rendering
仍未建立。TRAIN template=`55,000`、DEV role=`5,000`、selection=`1,000`，calibration/protected/test/sealed=`0`。

运行边界保持：CUDA:0、RTX 3060 Laptop、8.6、torch 2.5.1/CUDA 12.1、grid=`2048`、batch=`10`、modal projection、
`T=I_100` 与 shared `W_OLS` 均通过；optimizer/backward/optical update/retraining/Stage B/scale/prior/generation/
formal promotion 均未执行。report SHA-256=`93623cdc1d91f7418afd06ef49e53677c8f0c001801d8ae99c97aae738902e50`，
lineage SHA-256=`2ac4a8d23d5bc0228032597f94d541e12f7ebd8a60c501cb6b1313ec06c47aea`，input SHA-256=
`604cb8cef33e64e51a1e95a70fc5f3e3fe751849cbbdef0cddd1fd680f597b7b`。attempt-01/02 不变，不 retry；
`V4_GENERATOR_CONDITIONALITY_RESOLVED` 仍 unresolved。

## 当前 OLS class-direction transfer diagnosis attempt-01（2026-09-11）

用户独立授权 `DIAGNOSE_COMPLEX_OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_V1` 一次且仅一次，范围为
forward-only/no-update。当前 consumer HEAD=`054026ad3b44558b108888178ac2fb58aebb2f52`，branch 匹配；
历史 producer HEAD 与 consumer HEAD 按 `HASH_ON_TRUST_BOUNDARY_V2` 分离记录。新入口在 authority/runtime loading
阶段调用不存在的 `_load_evaluators` helper，尚未进入 TRAIN/DEV role reconstruction、encoder forward、OLS centroid
construction、generator forward 或 Smooth evaluation，故结果为 `INVALID/invalid_harness`，classification=
`INVALID_OLS_CLASS_DIRECTION_TRANSFER_UNEXPECTED_EXECUTION_ERROR`，不是四类科学分类。

本次 `diagnosis_executed=false`、scientific forward=`0`、TRAIN/DEV diagnostic reads=`0`、input 未写入；optimizer、
backward、optical update、training、V4 retry、Stage B、scale/prior/sampled generation、protected/test/sealed read
与 formal promotion 均为 NO。按 one-shot 规则不修复后重跑；Outer Gate
`OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_RESOLVED` 未作科学判定，V4
`V4_GENERATOR_CONDITIONALITY_RESOLVED` 继续 unresolved。

产物：[`report`](../../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_v1_attempt_01/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_report.v1.json)，
SHA-256=`25f2fc26a0aee688721e6ae9a2229db52a0b5d4b101cc0f9bbbf254e7dc2bae7`；[`lineage`](../../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_v1_attempt_01/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_lineage.v1.json)，
SHA-256=`bffda2cc3210fa7fad63d5d9df9af1971ea2037a8288831be97333f810fec73c`。执行时入口
[`script`](../../scripts/research/diagnose_lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_v1.py)
SHA-256=`95b85722fff33c09da83ac56024a7faa381c22f4cc24a83984d72fcf4a500697`。未改写任何既有 V3/V4 contract、
checkpoint、scientific report 或 INVALID history；未来若要修复并重开，必须重新授权并新建 attempt。

## 当前 OLS class-direction transfer diagnosis attempt-02（2026-09-11）

用户随后明确授权 `DIAGNOSE_COMPLEX_OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_V1` scientific attempt-02 一次且仅一次；
consumer HEAD=`a7cd8e082ad1b09c16c5c70fb38f0459cc583779`，入口已改用 canonical V4 scientific consumer evaluator loader。authority resolution
完成后，入口在 CUDA device preflight 的 complex/FFT probe 停止，底层 `nvrtc-builtins64_121.dll` 无法打开，结果为
`INVALID/invalid_harness`，没有科学分类。

本次 `diagnosis_executed=false`、scientific forward=`0`、TRAIN/DEV scientific reads=`0`、input 未写入；optimizer、backward、optical update、
training、V4 retry、Stage B、scale/prior/sampled generation、protected/test/sealed read 与 formal promotion 均为 NO。attempt-02 按
one-shot 规则消费且不重试；Outer Gate `OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_RESOLVED` 仍未作科学判定，
`V4_GENERATOR_CONDITIONALITY_RESOLVED` 继续 unresolved。

产物：[`report`](../../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_v1_attempt_02/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_report.v1.json)，
SHA-256=`e35b3f00de4b3dc48394a384c91627169a4c835809e2951211eee0ebd67243c2`；[`lineage`](../../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_v1_attempt_02/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_lineage.v1.json)，
SHA-256=`c70a8fef611e683da7e4b245f48b0daf95ac396f209608a2d041cca208c4cc64`。执行时入口工作树 SHA-256=`59e7a882e324378ab2011d940f979d7a7f24ff07139a19a9ab65b34da74f835d`；
attempt-01 及其修复历史保持 immutable。

## 当前 class-direction 运行时工程 readiness 核查（2026-09-11）

本轮没有重试 class-direction attempt-02，也没有执行 attempt-03。隔离复现显示 `_cuda_device_facts` 的具体失败操作是 CUDA complex
`torch.exp(complex_probe)`；complex allocation、`torch.polar` 与 complex FFT 均通过。真实 LightGen-aligned runtime 的 `[2,100]` 合成
complex OLS 前向在未修复环境中也因同一 `nvrtc-builtins64_121.dll` 失败，位置落在实际 generator complex path；故这是 CUDA runtime DLL 搜索路径
问题，不是非代表性的 device-facts microprobe。

当前环境的 `sys.prefix/bin` 已有 `nvrtc-builtins64_121.dll`。class-direction 入口现只在当前 Python 进程把该目录加入 `PATH`，不下载/替换 DLL、
不重装 PyTorch/CUDA；目录或 DLL 缺失时 fail-closed 为 `CUDA_RUNTIME_ENVIRONMENT_BLOCKER_CONFIRMED`。修复后按
authority resolution → CUDA facts → V4 step-300 checkpoint/runtime assembly → Frozen/Smooth evaluator load → synthetic smoke 顺序完成 readiness：
production grid=`2048`，真实 `W_OLS`、`T=I_100`、modal projection/re-emission、两层 generator 与 output relay 均执行；raw CMOS 为
`cuda:0/[2,1,200,200]/float32`，finite/nonconstant，runtime state unchanged，gradients none。

该 smoke 只使用纯合成 complex OLS，科学数据角色未打开；TRAIN/DEV/calibration/protected/test/sealed reads=`0`，scientific forward=`0`，
optimizer/backward/optical update=`0`，并在科学数据入口前停止。新增 focused regression 覆盖环境路径、缺失 DLL fail-closed、NVRTC runtime failure blocker
映射和 readiness stop；focused tests=`8 passed`，`py_compile`、Ruff、`git diff --check` 通过。此结果仅为工程 readiness，不是科学 class-direction 结果，
不改变 `OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_RESOLVED` 未判定、`V4_GENERATOR_CONDITIONALITY_RESOLVED` unresolved、training/Stage B/
generation/formal promotion 关闭的状态。

本轮从 HEAD=`a7cd8e082ad1b09c16c5c70fb38f0459cc583779` 完成 focused commit
`9f239d9dd1971872a7d5b2345ae11130574d48e4`（`research: add CUDA runtime readiness smoke`）。既有状态文档修改与 7 个此前未跟踪研究脚本仍保持未提交，未被本轮提交或清理。

## 当前 OLS class-direction transfer diagnosis attempt-03（2026-09-11）

用户独立明确授权一次且仅一次
`DIAGNOSE_COMPLEX_OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_V1_ATTEMPT_03`。在旧入口 self-binding
`a7cd8e…` 与授权当前 consumer HEAD 不一致的情况下，仅将入口自绑定重绑定到当前 HEAD=
`9f239d9dd1971872a7d5b2345ae11130574d48e4`；没有改变冻结 intervention、数据角色、统计或 acceptance 规则。
authority resolution、CUDA facts、V4 step-300 runtime/evaluator assembly 与 production-grid synthetic readiness 通过后，
科学阶段执行了 `5,800` 次 forward（TRAIN encoder=`5,500`，DEV canonical/replay/shifted 路径已进入）。

入口随后在 raw-response summary 计算中调用 V4 helper 的不存在属性 `_raw_summary`，错误为
`AttributeError: module 'lightgen_v4_conditionality_helper_for_class_direction' has no attribute '_raw_summary'`。
因此结果为 `INVALID/invalid_harness`，classification=`INVALID_OLS_CLASS_DIRECTION_TRANSFER_UNEXPECTED_EXECUTION_ERROR`；
这是 partial scientific execution 的 harness 失败，不构成四类科学分类。没有可用的 raw/template/Smooth summary、bootstrap、十个 pair summary
或 broad-vs-pair-dominated 结论；Outer Gate=`OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_RESOLVED` 未判定。

不可变产物：

- [`input`](../../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_v1_attempt_03/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_input.v1.json)，SHA-256=`7b73020e355badbc279b2b1edfe346bdaef6019620bf5f2dc467b789dd631841`；
- [`report`](../../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_v1_attempt_03/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_report.v1.json)，SHA-256=`a0b1032428c78c9f02b7db005732886186c1f68d3335c35602d1d01d2f6c025c`；
- [`lineage`](../../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_v1_attempt_03/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_lineage.v1.json)，SHA-256=`03eb6a38d010a59de06ad18274ef186fc847c8fe94deb6553646389fc28f4876`。

Input role boundary: source=`60,000` loaded、TRAIN=`55,000`、DEV role=`5,000`、fixed selection=`1,000`；calibration/protected/official test/sealed=`0`。
Runtime target was CUDA:0 / RTX 3060 Laptop / capability=`8.6` / PyTorch=`2.5.1` / CUDA build=`12.1` / grid=`2048`，with frozen
`T=I_100`、shared `W_OLS`、modal projection/re-emission and V4 generator. `optimizer_created=false`、`backward_executed=false`、
`optical_update_executed=false`、`training_executed=false`；Stage B、scale/prior/sampled generation、protected/test/sealed read 与 formal promotion 均未执行。

attempt-03 已按 one-shot 规则消费，不重试；`training_authorized=false`、`stage_b_authorized=false`、`generation_readiness=not_ready`、
`formal_promotion=false` 保持，`V4_GENERATOR_CONDITIONALITY_RESOLVED` 仍 unresolved。任何修复/重开必须重新定义 bounded contract、取得独立授权并使用新 attempt；
本次 partial forward 不能授权训练、Stage B、generation 或 formal promotion。

## 当前交接结果：OLS class-direction transfer harness-failure replication（2026-09-11）

用户独立明确授权
`REPLICATE_COMPLEX_OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_AFTER_HARNESS_FAILURE_V1`，仅执行一次冻结语义的
forward-only/no-update replication。它不是 attempt-03 retry；attempt-03 的 `INVALID/invalid_harness`、scientific forward=`5800` 与
`ATTEMPT03_NO_FORWARD_RECOVERY_UNAVAILABLE` 保持 immutable。replication 从稳定化 runner consumer HEAD=
`9a72e0bcf2de12ab9a0caa7151f273de76768dc7` 执行，starting/ending HEAD 相同，历史 attempt-01/02/03 未改写。

唯一结果为 `VALID_COMPLETED`，classification=`OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_SUPPORTED`，Outer Gate=
`OLS_CLASS_DIRECTION_TO_OPTICAL_OUTPUT_TRANSFER_RESOLVED`。前置 authority resolution、canonical evaluator loader、CUDA readiness、真实
`2048` production-grid synthetic LightGen smoke 与 fresh write-once output path 均通过；设备为 CUDA:0、RTX 3060 Laptop、capability=`8.6`、
PyTorch=`2.5.1`、CUDA build=`12.1`。固定 runtime 为 modal projection/re-emission、`T=I_100`、shared `W_OLS` 与 V4 step-300 generator。

数据边界为 source=`60,000` loaded、TRAIN=`55,000`、DEV role=`5,000`、fixed selection=`1,000`（每类 `100`）；calibration/protected/official
test/sealed 均为 `0`。总 scientific forward=`5,800`，其中 TRAIN encoder=`5,500`、DEV canonical/replay/shifted=`100/100/100`，Smooth evaluations=`200`。
optimizer/backward/optical update/training/Stage B/scale/prior/sampled generation/protected/test/sealed/formal promotion 均为 NO。

冻结 intervention 为 `t=(y+1) mod 10`、`z_shift=z-mu_y+mu_t`、`alpha=1.0`、无 renormalization。z、z_shift、centroid direction 与 raw outputs 均 finite；
raw absolute RMS change mean/median=`0.1127222995/0.1075329755`，relative mean/median=`0.8655350143/0.8446657972`，raw shape=`[1000,1,200,200]`，
structured/nonconstant=true。Template Delta mean/median=`0.0035271140/0.0021154392`，95% CI=`[0.0029781869,0.0040915711]`，fraction >0=`0.600`；
Smooth Delta mean/median=`0.8398663402/1.4927580953`，95% CI=`[0.6562327564,1.0271191061]`，fraction >0=`0.605`；两者 primary pass 均为 true。
十个 pair 中 template 与 Smooth 均有 `7/10` 正向 pair means，broadness=`broad`、非 pair-dominated；完整 pair 表见 [当前状态](../CURRENT_STATUS.md)。

Sufficient observations 已在 bootstrap/classification/report assembly 前 write-once 写入并回读验证：
[`artifact`](../../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_v1_attempt_01/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_sufficient_observations.v1.pt)，
SHA-256=`0cce9236ef8ad79cb4898b4ed9c3dec121d50e9060d7dffdbf1bcb4274f0e87e`；1000 samples、每类 100、selection identity、target rule、12 个 tensor 的
schema/shape/set/hash/reload 均通过。完整 report/lineage 为：
[`report`](../../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_v1_attempt_01/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_report.v1.json)，
SHA-256=`798e5af37f9d5e477fc395bd8230d17f25a20e56b0371f7376cfd1fbb7189883`；
[`lineage`](../../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_v1_attempt_01/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_lineage.v1.json)，
SHA-256=`d81c3829c8ae911d4aa0fd7620fca06d6dbecc16b3ee0292e024af5a76fa4496`。显式记录 replication identity、attempt-03 关系、历史事实及
FACT / MODEL_ASSUMPTION / DESIGN_DECISION 的 write-once envelope 为：
[`replication report`](../../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_v1_attempt_01/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_report.v1.json)，
SHA-256=`f5b1b252b5815578fd09367258df66e7ba8f3ea008fc889cb30c23ec8d18c249`；
[`replication lineage`](../../outputs/20260911_v4/runs/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_v1_attempt_01/lightgen_2d_fixed_ols_class_direction_to_optical_output_transfer_replication_lineage.v1.json)，
SHA-256=`7a75ddb6008a65057c86e658104b6ccf61b14bd9091382f2101d55dae70440e7`。

FACT：上述数据边界、forward count、finite/shape/raw、Delta、bootstrap、pair 与治理标志是本次 measured observations；attempt-03 是 post-forward harness INVALID，且不可由其旧 artifacts recovery。
MODEL_ASSUMPTION：两类 evidence 的 broad 正向响应支持 class-direction transfer interpretation，但不证明 generator capacity、V4 conditionality、语义生成或 Stage-A/Stage-B eligibility。
DESIGN_DECISION：固定 `t=(y+1) mod 10`、`alpha=1`、centroid replacement、full complex `C^100` gauge、无 renormalization 与原统计/分类规则。
V4 `V4_GENERATOR_CONDITIONALITY_RESOLVED` 仍 unresolved；本 replication 不授权 training、Stage B、generation、scale extraction 或 formal promotion。

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
attempt-03 至 attempt-06 的 harness 修复 artifacts 也未覆盖。本段末尾“下一动作是
`EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V2`”仅是该 preflight 时点的历史记录；V2 Stage-A attempt-02 已完成，当前动作以顶部
V3 objective-correction contract 为准，且不得自动启动。

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

> **Scope:** `HISTORICAL_RECORD` / `SUPERSEDED_CURRENT_TEXT`。以下 details 只保留其原记录时点的 provenance；其中的“当前”“下一步”和 flags 不得覆盖本页顶部 2026-09-18 current block，也不构成当前执行授权。

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

> **Scope:** `HISTORICAL_CLOSED` / `SUPERSEDED` for the project-level current route. The closeout facts remain valid for the Scaled-Chen line; its “current phase” and “next scientific task” are historical as of 2026-09-03 and are superseded by the M0 authority at the top of this document.

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
