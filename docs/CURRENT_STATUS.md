# 当前项目状态

更新时间：2026-10-14。执行规则以[根 AGENTS.md](../AGENTS.md)为准。三项相关工作当前阶段停点见[收尾报告](status/lightgen_three_workstreams_closeout_20261014.md)：身份定位为混合证据，OLS/G3 v5在限定系统内完成，Generator恢复验收仍为`INVALID_HARNESS / INCOMPLETE`。Gaussian/OLS v6相位优化已在另一独立授权会话中开始；该会话结果不在本页上述v5结论内。其余测试数量保留为对应历史研究或工程轮次的记录。

## 2026-10-14 前景收益与数字身份/笔画定位：两类人口方向相反，混合证据

- **结果与状态：** 保存证据消费状态`VALID_DESCRIPTIVE_EVIDENCE / diagnostic-only`；原训练三种子前景收益判定`FOREGROUND_GAIN_REPEATED_ALL_THREE_SEEDS`保持不变。本轮消费17,500行分析记录，重建7,500个A/B键，2,070行混淆汇总与已保存分析逐项相同；固定选出230对（144层中16层为空），不使用图册比例估计总体率。
- **科学认识：** 全人口E_D在经验DEV为A 2473/3000→B 2518/3000（净+45，三种子分别+16/+17/+12）；条件采样为3596/4500→3526/4500（净−70，三种子分别−20/−23/−27）。DEV较大退化类别为1（−23）、7（−14），改善为8（+34）、9（+26）；条件采样类别1（−95）、7（−33）退化，类别8（+37）、9（+12）改善。新发混淆示例：DEV 1→8为17对、7→3为15对；采样1→3为60对、7→9为23对；两臂常见持续错误包括DEV 9→4（91对）及采样9→4（139对）。条件采样类别1是最集中的代理退化方向。
- **目标缺口：** raw积分强度中位B/A约0.988；固定选样把B强度匹配到A后，生产视图平均绝对差DEV 0.010372→0.010226、采样0.009744→0.009594，230/230对均缩小，但单位总强度形态TV仍约0.1146/0.1139。当前没有对反事实能量匹配图运行E_D，不能判定该呈现收缩是否改变类别判断；结构/读出解释仍未区分。此处也没有人眼识读、TEST泛化或模型采用证据。
- **工程完成与验证：** [完整收尾报告](../outputs/20261014_m0_generator_identity_stroke_localization_v1/runs/attempt_02/report.md)、[机器验证](../outputs/20261014_m0_generator_identity_stroke_localization_v1/runs/attempt_02/validation.json)、[分层图册](../outputs/20261014_m0_generator_identity_stroke_localization_v1/runs/attempt_02/review_atlas.html)及[AutoDL回传收据](../outputs/20261014_m0_generator_identity_stroke_localization_v1/runs/autodl_backup_cloud_attempt_01/transfer_receipt.json)已保存。Windows CPU完整消费与AutoDL执行结果一致：460个端点、190个raw块、DEV Q_fg独立重算226次；原始视图与logits/余量逐项复核，无零强度样本。AutoDL预检与执行均退出0，遥测采样2次、GPU利用率0、采集失败0。聚焦测试5 passed，Ruff与Python编译通过；更新数、模型/评价器前向、新采样和人眼评分均为0。
- **决策影响与授权边界：** 将身份变化定为`MIXED_OR_UNRESOLVED`。唯一建议是对相同230对固定输出只改变B的标量强度至A的积分强度，再用冻结E_D比较转移和身份余量；此建议尚未执行，也不授权训练、采样、人眼评分、TEST或采用。

## 2026-10-14 Gaussian/OLS 窗口与维度功率复核 v5：扩展窗口通过，响应集中于少数通道

- **结果与状态：** AutoDL任务退出码0，状态`VALID_COMPLETED`。3072/4096/6144网格、共同4.096 mm ROI的9项生产complex64比较全部低于原`1e-4`门槛，最大对齐复场误差`6.01436e-5`；complex128最大`6.01410e-5`。据此完成3072网格零更新响应谱和功率测量。v4原2048窗口`G3_FAIL`保留历史，没有被改写。
- **科学认识：** 125 μm阵列零相位时，2×2及更大偶数阵列`N_eff≈4`且总耦合约Gaussian 3.751%、LP01 4.794%；增加偶数阵列通道数未提高单场有效维数。奇数阵列因光轴中心fiber位置不同，`N_eff≈1.02–1.04`，说明奇偶几何对齐与通道数量混杂。10×10四种相对阈值下均只有4/100通道有效。固定相位输入响应谱在10×10、1e-3阈值下，Gaussian实幅度有效秩为8、LP01为32；此数学谱对模式和阈值敏感，没有噪声/探测器数据，不能当作实验可用维数。
- **目标缺口：** phase-only优化尚未运行，K/H矩阵只验证固定相位下输入振幅响应；不能推断外圈fiber经相位优化可达性、最终OLS维度、非线性收益或生成质量。pitch/MFD等仍未测量。Gaussian→decoder/camera、材料与生成训练未验证；旧Generator恢复验收和39/45工程更新计数保持独立不变。
- **工程完成与验证：** [v5报告](ols_window_dimension_power_v5.md)、[机器JSON](../artifacts/ols_window_dimension_power_v5.json)、[精确AutoDL回传包](../outputs/20261014_ols_window_dimension_power_v5/runs/autodl_backup_attempt_02)及[回传后审计](../outputs/20261014_ols_window_dimension_power_v5/runs/attempt_01/post_return_audit.json)已保存。预检PASS，1782条探针记录/5346项预测比较全部通过，最大相对误差`1.543e-15`。161个文件共1,412,737,073字节逐项大小核验通过；云端临时源码已恢复并逐字节核对。聚焦测试5 passed，Ruff和编译通过。正式训练、phase-only优化、数据/evaluator/FID、loss、材料前向和几何修改均为0。
- **决策影响与授权边界：** 当前证据足以支持单独规划固定系统上的phase-only诊断，目标需分开为最大吞吐、最大`N_eff`和均匀100通道交付，并继续同场Gaussian/LP01对照；奇偶中心对齐应作为显式候选因素。没有选出最终通道数或结构，本轮停止，不授权下一轮优化或训练。

## 2026-10-14 Gaussian/OLS 独立参考闭合 v4：参考有效，G3 仍 FAIL

- **结果与状态：** AutoDL 完整任务退出码0，独立CPU带限积分参考为`VALID`；原生产 Gate 保持`VALID_FAILED / G3_FAIL`。GL4 32768节点下最终两级及 GL8 交叉阶数在全部3相位×3点组通过`1e-7`收敛要求。生产complex64最大共同全局相位对齐复场误差`1.6014370095e-4`，complex128为`1.6014355370e-4`，原门槛`1e-4`未变。
- **科学认识：** 有效参考将结论推进到125个指定位置；生产与参考的误差随窗口增大而降低，但125点不能证明全ROI机制。G3完整ROI仍有效失败；周期边界与FFT频率采样未分离。本轮未证实生产实现新缺陷，也不据此判定Gaussian或光纤架构失败。
- **目标缺口：** 仍缺完整ROI G3通过或可解释其误差机制的证据；K/H响应矩阵、维度排序、Decoder/camera接口及物理标定未验证。旧生成checkpoint恢复验收独立待办，历史工程更新计数39/45。
- **工程完成与验证：** [v4报告](ols_g3_reference_closure_v4.md)、[机器证据](../artifacts/ols_g3_reference_closure_v4.json)、[完整AutoDL回传包](../outputs/20261014_ols_g3_reference_closure_v4/runs/autodl_backup_attempt_01)、[修正预检回传包](../outputs/20261014_ols_g3_reference_closure_v4/runs/autodl_backup_preflight_rechecks)、[回传后重读审计](../outputs/20261014_ols_g3_reference_closure_v4/runs/attempt_01/post_return_audit.json)。66+13个文件逐项回传与大小核验通过；18份原始积分归档重读通过；修正预检聚焦测试8 passed、编译/Ruff通过；临时云端源码逐项恢复。训练、优化、数据/evaluator/FID、loss、材料和几何更改均为0。
- **决策影响与授权边界：** G3仍阻断响应谱、维度排序和结构决策。建议另行设计原全ROI误差来源诊断，沿用原窗口、生产complex64和`1e-4`；本轮不授权该后继测量或训练。

## 历史：2026-10-14 Gaussian/OLS 第二段传播误差定位 v3（由v4补充）


- **结果与状态：** AutoDL任务退出码0，`VALID_FAILED / G3_FAIL`。保持原4.096/6.144/8.192 mm窗口、共同4.096 mm ROI及`1e-4`门槛；complex64最坏对齐复场误差`1.601436996e-4`，complex128为`1.601435537e-4`。最坏项为末层xy各1°、2048对4096。
- **科学认识：** 固定同一第二段入口后，窗口误差仍约`1.60e-4`；第一段ROI误差最大`7.79e-6`、200 μm aperture crop误差最大`2.01e-7`。complex64/128差异很小，坐标最大差仅`4.34e-19 m`。但CPU独立带限积分未达到预设收敛门槛，故只能定位到第二段窗口依赖，不能区分FFT频率采样与周期边界交互；没有找到本轮可证实的生产实现缺陷。
- **目标缺口：** G3尚未闭合，独立参考的25个ROI采样点未收敛；OLS响应矩阵/维度排序、训练和Gaussian→camera接口仍未验证。
- **工程完成与验证：** [本轮报告](ols_g3_second_hop_error_isolation_v3.md)、[机器证据](../artifacts/ols_g3_second_hop_error_isolation_v3.json)、[精确回传证据](../outputs/20261014_ols_g3_second_hop_error_isolation_v3/runs/autodl_backup_v3_evidence)已保存。云端预检通过，修复遗漏依赖后的回归47 passed；63项回传逐项核对，本地18个复场、37个中间场张量和36张误差图重读均通过。临时云源码已恢复。训练、相位优化、数据/evaluator/FID、loss、材料和几何更新均为0。
- **决策影响与授权边界：** 当前不支持维度测量或结构决策。建议先另行解决独立参考的收敛问题，再依原窗口、几何和门槛复核G3。本轮停止，不授权训练、OLS矩阵、材料或相机接口变更。

## 前序：2026-10-14 Gaussian/OLS 零更新收尾：G3 仍未通过

- **结果与状态：** `VALID_FAILED / G3_FAIL`。修复传播核相位精度后，complex64 生产路径共同ROI最坏全局相位对齐复场误差为`1.6014e-4`，仍高于原`1e-4`门槛；complex128为`1.6014e-4`。完整OLS响应矩阵和维度排序按门控未运行。
- **科学认识：** float32传播核错误已由独立参考与负例证实并在前序任务中修复。该轮中间比较提示窗口相关传播差异，但原因尚未由独立物理参考确认；最新v3将差异定位到固定入口后的第二段，独立积分未收敛，周期边界和频率采样仍未分离。逐fiber latent稳定不替代G3复场门槛。
- **目标缺口：** v3已把窗口差异定位到第二段，但还缺有效收敛的独立参考；Gaussian维度排序、训练和Gaussian→camera重建接口均未验证。
- **工程完成与验证：** [零更新收尾报告](ols_gaussian_zero_update_closeout_v2.md)、[机器证据](../artifacts/ols_gaussian_zero_update_closeout_v2.json)、[G3原始复场及精确回传目录](../outputs/20261014_ols_gaussian_zero_update_closeout_v2/runs/autodl_backup_attempt_09_raw)、[逐fiber读出](../outputs/20261014_ols_gaussian_zero_update_closeout_v2/runs/autodl_backup_attempt_10)。AutoDL退出码0，G3为有效失败；临时云源码已按备份逐字节恢复。训练/优化/数据/evaluator/FID/loss/几何改动均为0。
- **决策影响与授权边界：** 建议另行定位第二段传播的频率采样与周期边界影响，保持原G3口径；只有生产G3通过后再规划维度测量。本轮停止，不授权训练、维度矩阵、材料或相机接口变更。

## 2026-10-14 AutoDL连接修复：固定云端训练与长验证

- **结果与状态：** `VERIFIED_CONNECTIVITY_RESTORED`；同一Check在Codex受限沙箱内SSH255/DNS报错，在沙箱外退出0。已验证screen、CUDA和SCP完整链路。
- **科学认识：** 科学判断未改变；此前据沙箱错误推断本机或云端DNS/代理故障的结论撤回。
- **目标缺口：** 连接修复本身已完成；后续OLS逐通道latent补算和G3重测见本页更新条目，G3仍FAIL。
- **工程完成与验证：** [连接报告](autodl_connection_repair_v1.md)、[机器证据](../artifacts/autodl_connection_repair_v1.json)；后台任务退出0，CUDA求和符合523776，8文件SCP回传COMPLETE且本地重读通过。未改DNS、密钥、host key或云端源码。
- **决策影响与授权边界：** 后续训练和长数值验证固定AutoDL；本地准备/审阅/备份。根AGENTS明确SSH/SCP使用`require_escalated`，不再因沙箱错误要求用户修服务器；本轮科学更新0并停止。

## 2026-10-13 Gaussian S12 后续复核：G3 FAIL，维度测量受门控

- **结果与状态：** 用户已选定后续仿真用 Gaussian S12 作 OLS 主参考、同几何 LP01 作模式敏感性对照；正式默认模型和旧 checkpoint 不迁移。AutoDL 零更新复核为 `VALID_FAILED / G3_FAIL`：complex64 最坏对齐复场误差 `6.9225e-4`，complex128 为 `1.6014e-4`，门槛均为 `1e-4`。OLS 响应矩阵、秩和维度排序未运行。
- **科学认识：** 固定物理 ROI、同 dx、共同全局相位对齐后仍失败；高精度 residual 表明问题不只是 complex64 累积误差。误差随窗口的频率间距变化，支持有限窗口离散频率采样影响；global phase 不是主要解释。功率与 fiber latent 误差较小不替代复场 Gate。
- **目标缺口：** G3 数值收敛和 Gaussian→Decoder→camera 接口尚未闭合；本轮无 OLS 响应矩阵或新维度判断。逐通道复 latent 误差数组尚未补齐；2026-10-14已确认原连接报错源于沙箱隔离，沙箱外连接已恢复。fiber MFD、位置、功率和相机仍是未标定仿真假设。
- **工程完成与验证：** [复核报告](ols_gaussian_dimension_reaudit_v1.md)、[机器证据包](../artifacts/ols_gaussian_dimension_reaudit_v1.json)、[云端原始读出与回传收据](../outputs/20261013_ols_gaussian_dimension_reaudit_v1/runs/autodl_backup_attempt_01)已保存。原云端测量退出码0；18个复场读出均成功加载且有限；主证据 907,134,033 bytes，逐文件大小核对通过。追加的 `autodl_run -Action Check` 在上传前因 SSH 255/DNS 解析失败，逐通道复 latent 后处理没有启动；云端三份既有源文件已逐字节恢复。训练、数据读取、评价器/FID和loss修改均为0。
- **决策影响与授权边界：** 保留 Gaussian S12 主参考与 LP01 对照的用户选择；当前不支持 OLS 维度排序。使用已验证沙箱外AutoDL通道补齐已保存复场的逐通道读出，再设计 G3 数值收敛修复确认，之后才规划受控维度测量；该建议不授权训练、结构变更或loss修改。未来训练按物理功率前向、过程监控及结果报告保留功率，不新增功率损失；现有 `L_op` 和相机接口须在 Gaussian 训练前单独核验。

## 2026-10-13 compact OLS interface scaling：敏感性几何命中，实物候选未成立

- **结果与状态：** [compact OLS scaling 报告](compact_ols_architecture_scaling_report.md)与[结构化测量](../artifacts/compact_ols_scaling_results.json)为VALID_COMPLETED / SIMULATION_SCREEN_MET_SENSITIVITY_PITCH。280个计划扫描点与4个固定距离对照完成；59个phase-only优化结果通过筛选，但全部使用非参考fiber pitch，125 μm参考pitch为0个。
- **科学认识：** 当前200 μm aperture、125 μm pitch、z=1.5 mm零相位基线为eta_total 6.471%、N_eff≈4。只优化总耦合能升至55.645%，N_eff仍约4，说明总耦合不能代表100维通道覆盖。最小筛选命中为500 μm aperture、z=0.5 mm、pitch 50 μm，但50 μm pitch小于当前125 μm包层，属于敏感性仿真，不是可加工方案。
- **目标缺口：** 缺少真实fiber pitch/MFD/包层、relay/illumination尺寸与损耗测量；phase-only每个目标仅从零相位执行24步，并非全局上界。没有由本轮建立的100维参考几何最小接口，也没有制造或材料非线性结论。
- **工程完成与验证：** 云端预检通过、命令exit 0、结果VALID_COMPLETED；三种目标共20,448诊断优化步，正式训练更新、数据读取、evaluator调用、loss/生产几何改动均为0。telemetry 18条、采集失败0；聚焦测试5项、Ruff、compileall、差异检查通过。
- **决策影响与授权边界：** 建议先测量relay、illumination和fiber几何，再单独规划高精度无训练验证。当前仅授权本次诊断；正式训练、evaluator、loss修改、材料器件结论、生产几何变更和加工验收仍禁止。

## 2026-10-13 云训练1：attempt_02 工程诊断未闭环

- **结果与状态：** 此前三种子Generator-only科学训练仍为`VALID_COMPLETED / FOREGROUND_GAIN_REPEATED_ALL_THREE_SEEDS`，12,000次科学更新结论不变。本次工程诊断为`diagnostic-only / INVALID_HARNESS`：保留历史3次，本轮新增36次工程Adam更新，累计39/45；新预算还余6次，科学更新为0。云端独立恢复的边界状态逐项精确回读，但继续两步后的Generator终点超出预先冻结的同机连续运行包络，故不能宣称恢复通过。
- **科学认识：** 本次没有新科学测量。微批1通过双臂梯度等价；微批2、5、10至少一项梯度不等价，故推荐维持1。优化后微批1较旧生产方法测得约31.24%更快，但该性能结果不弥补恢复失败。
- **目标缺口：** SIGINT/SIGTERM保存演练、20行DEV及40行采样的生产保存回读、原完整raw判定重算均未执行。剩余6次不足以重做失败的三步恢复并完成两种信号恢复（至少还需9次），因此本轮按授权预算停止；完整恢复与微批切换验收未闭环。下次长训练前必须先定位恢复数值差异，并用另行明确的预算重新完成受影响验收。
- **工程完成与验证：** 执行器及维护脚本已部署，云端CLI/编译和输入核验通过。六轨迹18次、零lambda负控3次、连续重复12次及一次独立进程恢复3次已完成；恢复终点最大绝对差`4.8250e-5`、RMS差`1.3518e-6`，高于冻结包络`1.7881e-5`和`4.8921e-7`。资源采集11条、采集失败0。精确清单回传93项；收据和Windows文件逐项核对总计2,065,073字节，全部文件存在且大小匹配。validation汇总另记2,089,564字节，两者相差24,491字节，差异来源未定位并已单独记录。37个恢复/审计张量仅列入audit清单未传，因测量阶段未到达而没有新raw。旧三张图共342,195字节，仅复用历史图。Windows聚焦测试31项通过，Ruff、编译、差异检查通过。详细证据见[attempt_02验证](../outputs/20261012_autodl_training_workflow_improvement_v1/runs/attempt_01/engineering_attempt_02_validation.json)、[回传核对](../outputs/20261012_autodl_training_workflow_improvement_v1/runs/attempt_01/engineering_attempt_02_transfer_reconciliation.json)、[恢复分析](../outputs/20261012_autodl_training_workflow_improvement_v1/runs/attempt_01/engineering_attempt_02_recovery_analysis.json)及[收尾报告](../outputs/20261012_autodl_training_workflow_improvement_v1/runs/attempt_01/closeout.md)。
- **决策影响与授权边界：** 当前工程收尾不完整，不应把该入口标为训练`READY`。保留原始差异和容差，不增加Adam预算、不改科学结论。下一项科学主线仍建议先做类别混淆与关键笔画定位；这不授权训练、采样、评分、调权或模型采用。

## 2026-10-13 云训练1：保存证据数字身份定位完成

- **结果与状态：** 保存证据分析覆盖7000行DEV和10500行条件采样；父任务`FOREGROUND_GAIN_REPEATED_ALL_THREE_SEEDS`原判定保持不变。本轮科学更新、云端工程Adam更新、新采样和人工评分均为0。报告与验证见[本轮收尾](../outputs/20261013_m0_generator_cloud_digit_identity_saved_evidence_v1/runs/attempt_01/closeout.md)和[主线路线图](research/lightgen_mainline_roadmap_20261013.md)。
- **科学认识：** B臂保持三种子Q_fg收益，但类别2、4的E_D焦点正确数和身份余量略降；固定DEV目标的两类自身形态距离下降，类别分离方向混合。全十类身份表现存在明显类别权衡，raw CMOS积分强度在焦点DEV/采样约下降1.2%。这些均为描述性代理，不是人眼识读或采用证据。
- **目标缺口与下一步：** 云端真实进程退出后恢复、信号保存、资源采集及微批双臂性能尚未验证；42次工程预算已用3、余39。建议先定位类别混淆/关键笔画并设计一个单因素目标诊断，再决定是否执行160题识读方法试点。下一次长更新前必须完成对应云端恢复和生产预检。该建议不授权训练、评分或模型采用。
- **工程完成与验证：** 保存证据入口预检为`READY`；父完整raw读回记录复核了7000+10500行。现有CSV重新消费验证通过：7500个身份配对键、327项行到汇总比较、1500个焦点身份配对和1200行DEV焦点Q_fg与父raw判定一致。39项聚焦测试、Ruff及Python编译通过。因为attempt_01已有完整输出，入口按保护规则拒绝覆盖；本次报告刷新重读保存CSV，没有再次加载全部raw。
- **决策影响与授权边界：** 当前建议先定位类别混淆/关键笔画并设计单因素目标诊断，再决定是否执行160题识读方法试点。下一次长更新前必须完成对应云端恢复与生产预检。此路线不授权训练、评分或模型采用。

## 2026-10-13 物理功率与非线性审计完成：前向物理缺口仍在

- **结果与状态：** [14项审计报告](physical_power_audit.md)和[结构化JSON](../artifacts/physical_power_audit.json)已完成，`AUDIT_COMPLETED_WITH_PHYSICAL_GAPS`。现有2048补零窗口与复LP01 overlap已存在；绝对激光功率、每层材料透射、125 μm实际阵列映射和532 nm材料层尚未接入。独立诊断与参数扫描完成，正式训练、评价器和模型采用均未改。
- **科学认识：** 大传播网格宽4.096 mm，但耦合平面仍200 μm，不能容纳名义1.25 mm阵列。Gaussian无孔径守恒误差2.33×10⁻¹⁵；小窗口负控在守恒时仍发生周期卷回。1 mW/T=.889下520 W/cm²场景覆盖率为LP01 3%、Gaussian 4%；这些是未训练光束及假定数值材料平面的读数，不能解释MNIST收益或证明材料可行性。
- **目标缺口：** 端面/像面、pitch/MFD/倍率、功率/透射、fiber复transport及532 nm材料曲线/时间条件未知。现有相对L_op不能提供W；文献0.889、0.2493 mW和TaP 520 W/cm²只作条件参考，3×10⁷的Te@532 nm出处未核实。
- **工程完成与验证：** [最终诊断](../outputs/20261013_physical_power_audit_v1/runs/diagnostic-only_attempt_03/diagnostics.json)及[预算](../outputs/20261013_physical_power_audit_v1/runs/diagnostic-only_budget_02/report.json)保存200条预算、20,000条通道功率、450组覆盖分布、7,200条材料响应。聚焦测试14 passed；Ruff、AST和空白检查通过。开场Mapping缺导入随后在工作树补齐，非本任务改动；完整未训练入口探针现通过。未运行全套pytest。
- **决策影响与下一步：** 保留L_op，先为目标2+2闭合单次入口W标定与阵列参考面，再按A1–A4逐项接入；功率约束A5仅作候选。此为建议，未授权下轮正式模型修改、训练、采用或物理部署；本轮交付后停止。

## 2026-10-13 主线路线更新：attempt_02前快照，后续已更新

- 路线及证据边界见[主线路线图](research/lightgen_mainline_roadmap_20261013.md)。本次仅重读报告、汇总保存CSV并更新文档，无训练、模型前向、新采样或人审。
- 上轮三种子B臂类别2、4前景误差相对A低约18.1%、相对共同起点低约27.0%；正式结论不变。附属读数并不同向：DEV目标两类E_D正确率A为96%、B为95.5%；条件采样十类请求类别E_D正确率B低于A，尽管形态集合距离改善。它们提示下一项应检验数字身份收益，不能直接认定人眼更好或更差。
- 下节工程停止是当时执行状态，不是所有主线分析的前置禁令。42次工程预算已用3、余39；从头重跑完整42次验收会累计45，但零更新分析和测量方案设计可先推进。原工程验收仍未闭合；恢复必须在下次相关长更新前验证，微批优化仅在切换前验收。
- 建议顺序：保存证据身份定位→识读测量设计→按缺陷决定是否安排新训练→锁定候选后的TEST与采用评审。路线建议不新增科学更新、诊断预算或模型采用授权，历史报告不改写。

## 2026-10-13 云训练1复盘与工程收尾：attempt_02前状态快照，后续已更新

- **结果与状态：** 此前三种子科学训练已完成12,000次更新，科学报告仍为`VALID_COMPLETED / FOREGROUND_GAIN_REPEATED_ALL_THREE_SEEDS`。本轮工作流诊断为`diagnostic-only / INVALID_HARNESS`：首个云端检查在累计3次工程Adam更新后因参考梯度字段错位退出；科学更新为0。修复已部署，A/B两臂零更新生产梯度复核通过，但恢复、完整微批性能和保存读出验收仍未完成。
- **科学认识：** 本轮工程修复没有新增科学结果，也未改变先前前景收益结论。首轮云端失败是把重建梯度误作Q_fg梯度，不是对照目标的科学失败；未放宽容差。
- **目标缺口：** 42次工程更新上限已用3次，剩余39次；原失败进程没有保存连续A臂状态，无法在不重做至少3次更新的情况下形成原计划要求的五次重复包络。为守住预算，本轮停止，不将未完成的恢复演练、六轨迹、负控、20/40行保存回读、性能基准和终点判定重算标为通过。没有为这次诊断生成新图表。
- **工程完成与验证：** 修复`_production_reference_batch_gradient`的元组字段读取，并新增回归测试；聚焦测试35项通过，Ruff、Python编译和`git diff --check`通过。执行器已先备份再部署，云端哈希与本地一致，`py_compile`及模块方式`--help`通过。云端零更新探针A/B两臂均在`rtol=2e-5, atol=2e-7`内匹配，340,000个元素超差数均为0，模型/Adam/评价器状态不变。资源采集器记录4次、失败0次并正常停止。13个精确文件已回传，合计958,467字节，大小全部核对一致；完整报告见[本轮收尾](../outputs/20261012_autodl_training_workflow_improvement_v1/runs/attempt_01/closeout.md)。完整pytest历史轮次为75失败、7错误、2,350通过；这两个聚焦文件不在失败列表。
- **决策影响与下一步：** 工程工作流改进尚未验收完成。建议停止本任务内进一步Adam更新；若仍需完成恢复与测量保证，应另行设计能够覆盖丢失重复基线的工程诊断预算。此建议不授权追加科学训练、调权、模型采用或部署。云端三个指定文件的临时覆盖流程已记入[AutoDL操作说明](operations/autodl_workflow.md)，不改变常规Git源码管理规则。

## 2026-10-12 云训练1工作流完善：本地实现完成，云端诊断待源码同步（当时状态快照）

- **结果与状态：** 已实现新运行检查点轮换、文件级恢复诊断入口、微批候选验证、60秒资源采集器、精简回顾包与显式清单回传。此次没有启动云端诊断，没有执行42次工程Adam更新，也没有新增科学更新。
- **科学认识：** 本轮未改变此前三种子训练的任何科学判断。新增三张PNG和回顾包从已回传的`autodl_backup_foreground_multiseed_r9`重新核算；用途是回顾和交付，不构成新的训练或独立确认。
- **目标缺口：** 只读SSH已确认云端在线，GPU与显式输入存在。初始受限沙箱拒绝DNS解析，不能据此推断服务器掉线。远端执行器没有本地新增的工程检查、微批和恢复功能，两个维护脚本也不存在；远端执行器未跟踪，云端工作树有163项改动。GPU端真实恢复演练、六轨迹三步更新、微批等价与吞吐测量、60秒云端资源采集，以及基于新云端运行的一次清单回传/Windows读回均尚未执行。源码合同要求Git同步，本轮约束禁止提交/推送，故等待用户明确同步方式。review包不含raw像素块，不能独立复算像素级指标。
- **工程完成与验证：** CPU聚焦测试39项通过，Ruff、7个Python文件编译、两份PowerShell解析和`git diff --check`通过。实际审查清单选出40个文件、160,445,781字节；本地完整备份为601个文件、7,619,136,292字节。已在[新工程输出](../outputs/20261012_autodl_training_workflow_improvement_v1/runs/attempt_01)生成review包与三张PNG；包清单45项逐一读回无缺失、共160,791,596字节，Windows A4700起点依赖存在，PNG共342,195字节。没有删除或覆盖原证据。
- **决策影响与下一步：** 当前仅完成Windows源码和本地证据侧的工程改进，不表示云端准入或训练READY。明确源码同步方式后继续同一诊断任务，先核对显式输入，再执行不超过42次工程更新与实际回传验证；本节不授权追加科学训练、调权或采用模型。

## 2026-10-12 云训练1：固定前景目标三种子 DEV 确认完成

- **结果与状态：** 云端独立预检通过，六条 Generator-only 轨迹各完成 2,000 步，共 12,000 次科学更新；完整 DEV 保存读出回读后为 `VALID_COMPLETED / FOREGROUND_GAIN_REPEATED_ALL_THREE_SEEDS`。执行器及聚焦测试见[入口](../scripts/research/lightgen/execute_lightgen_2d_generator_cloud_foreground_dev_multiseed_v1.py)和[测试](../tests/research/lightgen/test_lightgen_2d_generator_cloud_foreground_dev_multiseed_v1.py)。
- **科学认识：** 三个种子的 Q_A / Q_B / Q0 分别为 0.389666 / 0.319155 / 0.437730、0.390118 / 0.319527 / 0.437730、0.389833 / 0.319422 / 0.437730；均满足预先固定的 `D < -τ` 且 `Q_B < Q0`。这支持固定起点、人口、lambda 和步数范围内的重复数值前景收益。
- **目标缺口：** DEV 曾用于评价器开发，不是全项目盲测。正式识读、采样可读性、新颖性、模型采用、Encoder 联合更新、物理部署和 TEST 验收仍未解决。
- **工程完成与验证：** 云端独立准入覆盖数值/梯度参考、真实更新、负控、恢复和隔离；最终原始块、逐行 CSV、判定与条件采样均完成保存证据回读。聚焦测试 14 项通过，Ruff、编译和 `git diff --check` 通过。完整本地备份位于[运行收据](../outputs/20261012_m0_generator_cloud_foreground_dev_multiseed_v1/runs/autodl_backup_foreground_multiseed_r9/transfer_receipt.json)，报告见[收尾报告](../outputs/20261012_m0_generator_cloud_foreground_dev_multiseed_v1/runs/autodl_backup_foreground_multiseed_r9/closeout.md)。
- **决策影响与下一步：** 训练建议 `NONE`、后续动作 `NONE`；本轮到此停止，不自动采用模型或追加训练。2026-10-11 冻结重放仍为 `REPLAY_MISMATCH_ENGINEERING_REVIEW_REQUIRED`，原容差保持不变，本轮不声称严格复现 Windows 历史结果。

## 2026-10-12 AutoDL 工程工作流

- [正式操作说明](operations/autodl_workflow.md)固定 VS Code Remote-SSH、云端解释器、`screen` 后台任务、状态/日志查看与关键结果回传；[启动/检查脚本](../scripts/maintenance/autodl_run.ps1)、[回传脚本](../scripts/maintenance/autodl_pull.ps1)只接受显式任务、运行目录和文件清单，不自动选训练入口、检查点或更新数。
- 工具链验收记录见[收尾报告](../outputs/20261012_autodl_workflow_finalize_v1/runs/attempt_01/closeout.md)。环境检查实际观察到AutoDL Python 3.12.3、PyTorch 2.5.1+cu124、CUDA可用、RTX 4080（32760 MiB）和screen 4.09；本轮只运行shell操作测试，科学更新为0。
- 工程链验收不表示训练 `READY`。此处“下一项训练仍待单独确定”描述工作流验收当时；之后另行批准的云训练1已按固定预算完成，见上节。A4700研究基线与B5400研究候选的历史决策未由工程验收改变；2026-10-11云端冻结重放仍为[`REPLAY_MISMATCH_ENGINEERING_REVIEW_REQUIRED`](../outputs/20261011_autodl_fixed_endpoint_replay_v1/runs/attempt_01/report.json)，差异来源未定位。

## 当前决策概览

- **总目标：** 验证 BOGT 联合目标能否经统一 Encoder→复 OLS→Generator 映射产生可辨识的 CMOS 数字图像。
- **架构复场可视化交付：** [小／大架构可视化与 7 页 PPT](../outputs/20261010_lightgen_architecture_visual_comparison_v1/runs/attempt_01/presentation/lightgen_architecture_comparison_7slides_v2.pptx)及[验证报告](../outputs/20261010_lightgen_architecture_visual_comparison_v1/runs/attempt_01/validation_report.md)已完成；只描述固定 canonical TRAIN 确认人口，不改变 A4700 基线或模型采用决策。
- **10 节点复场可视化交付：** [验证报告](../outputs/20261010_lightgen_best_endpoints_10node_visualization_v1/runs/attempt_01/validation_report.md)、[40 页图谱](../outputs/20261010_lightgen_best_endpoints_10node_visualization_v1/runs/attempt_01/figures/complex_field_atlas.pdf)和[8 页可编辑 PPT](../outputs/20261010_lightgen_best_endpoints_10node_visualization_v1/runs/attempt_01/presentation/lightgen_best_endpoints_10node_8slides_v1.pptx)已完成；固定 A1000 与 B5400 前景展示候选，在 20 行 canonical TRAIN 同输入样本上呈现 10 节点相位／振幅，不改变 A4700 基线或模型采用决策。
- **8 节点单图复场可视化：** [A1000 振幅](../outputs/20261010_lightgen_best_endpoints_8node_allinone_visualization_v1/runs/attempt_01/figures/small_A1000_amplitude.png)、[A1000 相位](../outputs/20261010_lightgen_best_endpoints_8node_allinone_visualization_v1/runs/attempt_01/figures/small_A1000_phase.png)、[B5400 振幅](../outputs/20261010_lightgen_best_endpoints_8node_allinone_visualization_v1/runs/attempt_01/figures/large_B5400_fg_amplitude.png)、[B5400 相位](../outputs/20261010_lightgen_best_endpoints_8node_allinone_visualization_v1/runs/attempt_01/figures/large_B5400_fg_phase.png)及[验证报告](../outputs/20261010_lightgen_best_endpoints_8node_allinone_visualization_v1/runs/attempt_01/validation_report.md)已完成；每图含 20 行 canonical TRAIN 同输入样本和 8 个主节点，不改变 A4700 基线或模型采用决策。
- **前一停止点——类别2、4残余形态与类别分离CPU定位：** [attempt_07报告](../outputs/20261010_m0_generator_digit_identity_localization_v1/runs/attempt_07/report.md)完成1500行经验解码与3000行固定联合先验A/B采样保存证据复盘。唯一主量28×28类别分离差`D_margin=0.00074795`，容差`3.0042e-7`，判读`CLASS_SEPARATION_INCREASED`；类别2、4分别改善`0.00019108`和`0.00130481`，但最接近竞争类仍分别为8和9。视图敏感性检查同向。旧盲判65/100与69/100仅作描述，审阅质控仍失败。本次无模型前向、更新、新采样或新增人审；保留A4700基线、B5400研究候选，`training_recommendation=NONE`、`next_action=NONE`。
- **前一停止点：** 固定联合先验下A5400/B5400采样与单人识读对照的[正式报告](../outputs/20261009_m0_generator_fixed_prior_sampled_readability_v1/runs/attempt_01/report.md)和[详细收尾分析](../outputs/20261009_m0_generator_fixed_prior_sampled_readability_v1/runs/attempt_01/closeout_analysis.md)已归档。270题评分完整，正控20/20、全黑10/10、B重复19/20通过，A重复13/20未达18/20；终局`REVIEW_INCONCLUSIVE / REVIEW_QUALITY_CHECK_FAILED`。最早展示答案为A65/100、B69/100，描述性主差4个百分点、精确p=0.34375。A4700研究基线不变。
- **分类代理效度实验一：** 已完成340项评分归档。真实目标50/50、全黑10/10通过内部检查，但隐藏重复仅21/30，低于27/30要求；终局为`REVIEW_INCONCLUSIVE / REVIEW_QUALITY_CHECK_FAILED`。主样本失配71/155（45.81%），有限总体95%区间为38.20%–53.54%，仅作描述性结果，不确认或否定代理效度。详见[实验一报告](../outputs/20261009_m0_generator_classifier_proxy_validity_v1/runs/attempt_01/report.md)和[实验二计划](../outputs/20261009_m0_generator_classifier_proxy_validity_v1/runs/attempt_01/experiment_02_plan.md)。
- **形态与能量实验二：** [实验报告](../outputs/20261009_m0_generator_shape_energy_counterfactual_v1/runs/attempt_01/report.md)确认主差`mean(Q_BA−Q_AA)=-0.0196597`，容差`7.7457e-6`；对称形态贡献`-0.0198670`，能量贡献`+0.0004897`，交互`-0.0004146`。300行确认人口的B/A积分强度比均值为0.95378。该数值反事实仅针对保存读出，不是物理光学实现，也不改变B6700不作为研究基线的既有决策。
- **跨端口联合结构实验三：** [attempt_04报告](../outputs/20261009_m0_generator_cross_port_joint_structure_v1/runs/attempt_04/report.md)为`VALID_COMPLETED / JOINT_PAIRING_SHAPE_SUPPORT`。保持同类别、同角色内每端口复系数边缘多重集的10次独立端口置换后，300行确认人口的十类等权未截断归一化形态距离由0.02673948升至0.02977328；主差`D_joint=0.00303380`，容差`7.9547e-7`。十个种子效应均为正，整行置换差约`-6.94e-18`。正式6000行冻结CUDA读出及保存证据回读完成；200行拟合人口单独汇总。该结果支持当前A4700解码器从经验OLS联合配对获益，不检验光学架构混合的必要性，也不证明新联合采样器收益。attempt_01–03工程失败保留，科学判定只消费有效attempt_04。
- **单人测量稳定性检查：** [100题盲判页](../outputs/20261009_m0_generator_measurement_stability_v1/runs/attempt_01/blind_review.html)、[正式评分报告](../outputs/20261009_m0_generator_measurement_stability_v1/runs/attempt_01/report.md)和[补充分析](../outputs/20261009_m0_generator_measurement_stability_v1/runs/attempt_01/closeout_analysis.md)已归档。条件采样与经验OLS各30张、隐藏重复20张、真实正控10张和全黑负控10张；真实数字10/10、全黑10/10、经验OLS重复9/10通过，条件采样重复8/10未达9/10，故为`COMPLETED / PILOT_QUALITY_CHECKS_FAILED`。三个重复变化都在数字与“无法辨认”之间；条件采样和经验OLS首次图像各14/30认可请求类别，只作单人小样本描述。本检查不改写实验一21/30的原质控失败，也不构成E_D效度判断。
- **冻结联合先验对照：** [有效attempt_02报告](../outputs/20261009_m0_generator_frozen_joint_prior_comparison_v1/runs/attempt_02/report.md)为`VALID_COMPLETED / JOINT_PRIOR_NUMERIC_IMPROVEMENT`。A4700冻结解码下，300行确认人口的形态集合距离分块均值为`0.03093698`、联合均值为`0.02920771`，主差`D_prior=-0.00172927`，容差`8.1874e-7`；五个预设种子的效应均为负。正式3000行CUDA前向及保存重读完成，独立NumPy重算与主差相差小于`4e-18`。补充分解显示输出间形态差异增加12.25%，而输出-目标交叉距离均值增加约`0.00089804`，所以集合距离改善不意味着每张图更接近目标。E_D总数增加39/1500，但类别0、1、4减少；数据均属canonical TRAIN，联合候选尚无人审。初次预检的DLL搜索路径故障与统计维度修复均留有记录；有效科学结论只消费attempt_02。
- **已有判断及边界：** A/B各追加2000步至Generator Adam step6700。确认人口上E_D为起点230/300、A243/300、B257/300；B相对A增加14张。总体Q相对A仅下降约5.1%，类别2、4分别下降约6.3%、5.0%，均未达到要求的10%；B整体CMOS积分能量比起点低约5.3%，低于95%保留线。均为canonical TRAIN结果，不代表独立泛化或人眼可读。
- **此前复盘：** [保存证据CPU复盘报告](../outputs/20261009_m0_generator_saved_evidence_cpu_review_v1/runs/attempt_04/report.md)与[逐行配对数据](../outputs/20261009_m0_generator_saved_evidence_cpu_review_v1/runs/attempt_04/paired_rows.csv)完成500行CPU重算，复现三个端点汇总、A/B逐行指标和父报告科学判定。A4700→B6700确认人口E_D正确性转移为错→对28行、对→错1行；259行分数差与Q共同改善，0行分数差改善而Q恶化；300行Q改善同时能量下降。数据仍属canonical TRAIN。attempt_01保留早期输出，其中逐行CSV遗漏A端并列计数；attempt_02补齐CSV但未包含新增的父预检/源读出核验项；attempt_03完成READY与源读出重放核验；attempt_04按最终入口重新生成，为当前权威复盘。[复盘入口](../scripts/research/lightgen/audit_lightgen_2d_generator_saved_evidence_v1.py)和[聚焦测试](../tests/research/lightgen/test_lightgen_2d_generator_saved_evidence_v1.py)已验证。
- **决策与停止点：** A4700研究基线不变。固定先验A5400/B5400的270题评分已归档，本阶段以审阅质控未通过收尾；没有确认B的单人相对识读收益。后续建议先定位类别2、4的残余识读缺口，另行规划后才执行。不会自动训练、调权、重拟合先验或采用B5400；独立人口、覆盖与新颖性仍未确认，`training_recommendation=NONE`、`next_action=NONE`。

## 2026-10-10 固定联合先验下 A5400/B5400 采样与单人识读对照（评分归档完成）

**结果与状态：** 任务`M0_GENERATOR_FIXED_PRIOR_SAMPLED_READABILITY_V1`完成3000行数值测量及270题单人评分归档，工程状态`VALID_COMPLETED`，科学终局`REVIEW_INCONCLUSIVE / REVIEW_QUALITY_CHECK_FAILED`。正控20/20、全黑10/10、B重复19/20通过，A重复13/20未达到18/20要求。原始提交、评分明细、主配对表、报告及`completed.json`均已保存；完成标记只表示本阶段结束，不表示识读收益通过。

**科学认识：** 100个同输入配对按预定题号最早展示答案计算：A认可65/100、B69/100，双方认可62对、仅B7对、仅A3对、双方不认可28对；`D_h=0.04`，精确McNemar `p=0.34375`。因A质控失败仅作描述，概率本身也未达到0.05。类别2、4合计A3/20、B4/20，类别4仍有4→9混淆；不能把父训练15.61%的Q_fg收益写为已转化的识读收益。数值辅助28×28集合距离仍为A`0.02666559`、B`0.02296276`，差`-0.00370284`，五种子同向；交叉距离下降伴随输出内部差异减少。E_D全库为1202/1500与1214/1500，仅作辅助。所有目标及拟合人口仍属canonical TRAIN。

**目标缺口：** A端点重复识读稳定性及B相对A的识读收益仍未确认。B在20对上的稳定性通过不推广为总体稳定性。实验一分类代理效度及100题稳定性检查的历史终局不变；类别2、4的数字身份缺陷、覆盖、新颖性、独立人口与跨训练轨迹重复性尚待独立设计。

**工程完成与验证：** [正式报告](../outputs/20261009_m0_generator_fixed_prior_sampled_readability_v1/runs/attempt_01/report.json)、[收尾分析](../outputs/20261009_m0_generator_fixed_prior_sampled_readability_v1/runs/attempt_01/closeout_analysis.md)及[独立评分核验](../outputs/20261009_m0_generator_fixed_prior_sampled_readability_v1/runs/attempt_01/analysis/independent_rating_validation.json)保存270行原始回答、270行审阅明细与100行主配对表。重读确认提交副本逐字节一致、四项质控及四格计数一致、精确概率为11/32、报告与完成标记一致。修正了首次答案消费：11个B副本比标记primary的条目更早显示，1对答案不同，主计数从B68修正为69；原回答与映射未改，终局不变。CSV到最终判读的反例在修正前失败、修正后聚焦测试10项通过，Ruff、编译与差异检查通过；本轮无新CUDA前向或训练，未重复全套pytest。90行READY及3000行冻结测量的历史证据、两份工程错误记录继续保留。

**决策影响与下一步：** 本阶段已结束。B5400保留为有数值收益的研究候选，不作为已确认可读性改进的模型采用。建议后续先在保存证据上定位类别2、4及原始形态与生产显示之间的关系，再单独规划新样本识读或训练方案。建议不是执行授权；本轮不续训、不调权、不重拟合先验、不自动追加人审，A4700研究基线不变，`training_recommendation=NONE`、`next_action=NONE`。

## 2026-10-09 A4700前景目标单因素训练对照

任务`M0_GENERATOR_FOREGROUND_OBJECTIVE_TRAINING_COMPARISON_V1`有效完成，状态为`VALID_COMPLETED / FOREGROUND_TRAINING_RELATIVE_IMPROVEMENT`。在共同A4700模型、Adam与RNG状态上，对照臂保留原重建目标，候选臂只增加固定`lambda=0.11159457624315472`的归一化Q_fg项；两臂均只更新Generator，各追加700步至A5400/B5400。拟合人口为200行类平衡批次，确认人口300行未参与更新、调权或终点选择；两者均属canonical TRAIN。

唯一主量为类别2、4确认人口各30行等权Q_fg差。A4700、A5400、B5400均值分别为`0.44411628`、`0.42808681`、`0.36127857`；`D_train=-0.06680824`，预设容差`8.76174e-6`。相对A5400，B5400下降约15.61%；两类分别下降约15.02%与16.20%。对照臂也较起点下降约3.61%，说明不能把全部变化归因于前景附加项，但同起点同更新数比较支持该固定候选在本范围内有相对训练收益。两臂平均Q_bg、raw MSE与生产Q_28亦下降；焦点两类SSIM提高。辅助E_D正确数为A4700 230/300、A5400 237/300、B5400 242/300，仅是评价器读数，不替代人眼识读。

正式测量共3500行、七种情景各500行；READY预检包括保存基线重放、训练梯度与Q_fg独立导数核对、A/B真实三步更新、每臂保存恢复比较、零lambda负控及冻结评价器状态。保存读出块与逐行CSV独立重算后复现同一主差和分类。首个正式进程的NVRTC DLL搜索路径故障发生在首个更新之前，修复后重新执行；失败记录保留。聚焦测试16项通过，变更入口/测试Ruff及AST编译通过；本轮没有修改共享光学路径。

本结果只支持固定A4700起点、lambda、训练规则和700步范围内的相对Q_fg收益；每臂只有一条预设批次/RNG轨迹，未估计跨训练种子的波动。它不证明人眼可读性、分类代理效度、独立泛化、采样覆盖或模型采用。实验一质控失败和100题单人稳定性检查未通过的终局不变。保留A4700研究基线；建议另行规划冻结先验采样对照、关键盲判和独立人口确认，但它们不在本次授权内，保持`training_recommendation=NONE`、`next_action=NONE`。

## 2026-10-09 冻结A4700前景目标梯度核验

任务`M0_GENERATOR_FOREGROUND_OBJECTIVE_GRADIENT_V1`有效完成，唯一主分类为`CANDIDATE_LOCAL_RELATIVE_IMPROVEMENT`。仅用attempt_03原200行拟合人口的20个固定批次确定`lambda=0.1115945762`；在A4700冻结点构造单位RMS Generator相位下降方向，候选方向保留原生产重建、光功率和规范化E_S/E_I/E_C梯度，并加入`BETA*lambda*grad(Q_fg)`。主分析为原300行确认人口中类别2、4各30行。

固定`epsilon=1e-3` rad RMS的单行相位探针下，Q_fg均值为基线`0.44411628`、原方向`0.44224931`、候选方向`0.43879953`；主差`D_local=-0.00344978`，容差`9.0450e-6`。类别2与类别4均有相对收益；候选相对基线降低约`0.00531675`。这是冻结点数值方向响应，不是单因素训练结果。类别、角色与五种读出情景的raw MSE、Q_fg/Q_bg、SSIM、积分强度、截断率及E_D辅助汇总保存在报告和CSV中。

attempt_04完成500行×5情景共2500行CUDA读出，并回读raw分块和CSV独立复算。A4700保存基线使用父生产的10行微批重放；方向探针用单行微批，两方向保持相同微批路径。解析图像梯度、相位中心差分和两尺度导数核对通过，模型/评价器/优化器状态不变，模型与优化器更新均为0。attempt_01因NVRTC DLL搜索路径失败，attempt_02因单行raw与原10行微批基线比较失败，attempt_03因预检汇总代码异常结束；均保留为工程失败，只有attempt_04消费科学判读。增加候选梯度组合、前景收益伴随背景代价、确认人口筛选、源行/生产视图拒绝及扰动异常恢复回归后，聚焦测试11项通过；Ruff、Python编译及差异检查通过，完整测试套件未运行。

在该梯度诊断的停止点，结果仅支持单独设计后续训练。其后经另行授权完成的单因素训练对照见上方当前停止点；本节不改变梯度测量的历史结论。人眼可读性和独立人口泛化仍未解决；实验一代理效度及稳定性质控失败终局不变。

## 2026-10-09 前景形态保存证据定位

任务`M0_GENERATOR_FOREGROUND_SHAPE_LOCALIZATION_V1`已读取能量目标对照attempt_03的A4700、A6700、B6700三个端点和全部500行保存读出。300行确认人口为主分析，200行拟合人口单独汇总；全部属于canonical TRAIN。唯一主量为类别2、4目标前景Q的等类加权差`D_fg=-0.03037272`，数值容差`8.5318e-6`，终局`FOREGROUND_SHAPE_ERROR_REDUCED`。类别2差`-0.03161772`（A6700 0.420730→B6700 0.389112），类别4差`-0.02912773`（0.412449→0.383322）；对应相对下降约7.51%与7.06%。因此B端点的归一化形态收益确实包含目标前景区域。

解释仍有指标分歧：两类等权的前景+背景raw Q由0.51245降至0.47989，背景Q也下降；但前景区域raw MSE均值约升4.34%，其对全图raw MSE的面积加权贡献约升4.23%，整体raw MSE约升3.43%。归一化前景过量质量下降约22.2%，不足质量略升约1.9%，背景生成质量增加约11.6%。28×28 Q约降5.68%，SSIM下降约0.0166；E_D在这60行仍为57/60正确、无端点变化。B/A积分强度约低4.79%，截断像素比例约由0.244%降至0.097%。这些量表明前景归一化分布指标改善与原始强度误差、SSIM并非同向，不能称作整体图像质量或可读性改善。

最终产物见[五项报告](../outputs/20261009_m0_generator_foreground_shape_localization_v1/runs/attempt_01/report.md)、[机器报告](../outputs/20261009_m0_generator_foreground_shape_localization_v1/runs/attempt_01/report.json)、[500行配对CSV](../outputs/20261009_m0_generator_foreground_shape_localization_v1/runs/attempt_01/paired_rows.csv)、[逐类逐角色汇总](../outputs/20261009_m0_generator_foreground_shape_localization_v1/runs/attempt_01/class_summary.csv)、[类别2/4全部60行图库](../outputs/20261009_m0_generator_foreground_shape_localization_v1/runs/attempt_01/focus_gallery.html)、[验证记录](../outputs/20261009_m0_generator_foreground_shape_localization_v1/runs/attempt_01/validation.json)和[执行入口](../scripts/research/lightgen/audit_lightgen_2d_generator_foreground_shape_localization_v1.py)。父保存指标与生产视图复现、10行代表性CPU预检、全部500行测量、两项逐行恒等式及CSV回读独立判读通过；定位图库按源行号排列，使用保存28×28视图、固定灰度与8倍最近邻放大。报告渲染器的字段引用缺陷在最终归档前修复并增加回归；完整测量数据从已落盘CSV和机器报告复核完成。聚焦测试12项通过，Ruff、Python编译和差异检查通过；没有模型更新、光学/评价器前向、梯度、采样或人工评分。

该定位不证明梯度因果、最佳损失权重、数字拓扑或人眼可读性，也不补足实验一质控和独立泛化。当时建议的生产梯度核验及单因素训练对照已在后续分别获准完成，当前训练结论见本页最新停止点；其他目标修改、B6700采用、采样或新增人审仍需单独授权。当前`training_recommendation=NONE`、`next_action=NONE`。

## 2026-10-09 保存证据CPU复盘

任务`M0_GENERATOR_SAVED_EVIDENCE_CPU_REVIEW_V1`消费能量相容目标对照attempt_03的三个500行读出、A/B逐行指标、人口清单、父报告与两臂训练历史。新入口只读CPU保存张量，无模型更新、光学前向、评价器前向、梯度、采样或人审。拟合200行和确认300行分别汇总，A4700→A6700、A4700→B6700、A6700→B6700按源样本行号配对；报告同时保留十类统计、四评价器读数和每20批次的训练日志汇总。

确认人口A4700→B6700中，E_D正确性为错→对28行、对→错1行；259/300行E_D分数差提高且Q改善，0/300行分数差提高而Q恶化。300/300行Q改善同时积分能量下降，能量变化中位数为`-9.5378e-10`。这些是固定端点的TRAIN人口关联，不能证明梯度因果、人眼可读性、独立泛化或采样生成收益。总体模式削弱广泛的分类—笔画反向竞争解释，后续若继续应先聚焦亮度归一化形态与绝对输出强度的联系。

最终完整输出为[attempt_04](../outputs/20261009_m0_generator_saved_evidence_cpu_review_v1/runs/attempt_04)：`report.md`、`summary.json`、`paired_rows.csv`及`validation.json`。三端点指标重算复现父报告汇总；A/B逐行指标和父科学决策按既有生产判定器成功重放；READY预检、A4700源读出回放、数据身份、物理坐标、2000条/臂更新历史均核验通过。聚焦测试10项通过，Ruff、AST编译检查与`git diff --check`通过，未运行全套pytest。attempt_01保留早期CSV计数漏项；attempt_02补齐CSV但未包含新增的READY与源回放检查；attempt_03已通过这些数据核验；attempt_04依据最终入口重新生成，不覆盖历史产物。保持`training_recommendation=NONE`、`next_action=NONE`，无后继实验授权。

## 2026-10-09 能量相容配对目标对照

任务`M0_GENERATOR_ENERGY_MATCHED_TARGET_COMPARISON_V1`固定A4700大Generator及其Adam状态，两臂均保留三评价器真实类别CE、BOGT和共同的raw MSE＋0.1 SSIM重建规则；A使用原目标，B只在配对重建目标中使用拟合人口固定的逐图能量系数`s_i`。系数只作用于重建目标，均值约0.584、范围0.438–0.717；未缩放Encoder输入、OLS、CMOS或评价器图像。两臂各追加2000步，最终Generator Adam均为6700。确认人口为300行canonical TRAIN，不是独立泛化集。

最终报告为`VALID_COMPLETED / MIXED_OR_INSUFFICIENT`。A4700、A6700、B6700在新300行上的E_D正确数分别为230/300（76.67%）、243/300（81.00%）、257/300（85.67%）；B较A多14张，十类保护条件通过。共同原目标Q为0.44996、0.37729、0.35791；B相对A只下降5.14%，类别2下降6.28%、类别4下降5.05%，均未达到预设至少10%的条件，虽均比A4700起点好。B的平均CMOS积分强度为`1.7947e-8`，比A6700的`1.8827e-8`低4.67%，比A4700的`1.8951e-8`低5.29%；能量保留条件失败。B的raw MSE和原目标SSIM也比A6700差（0.05309对0.05112；0.3472对0.3648），但相对起点改善；截断比例降低至0.0674%，各输出有限且无零图/恒定图。该结果显示自动形状指标、亮度和分类读数之间存在取舍，不构成人眼质量收益证据。

有效同入口CUDA预检READY后完成正式两臂训练和三个500行端点读出。预检覆盖源重放、四评价器真实目标正控、A/A重复性、A/B真实更新、配对目标与负控、梯度参考、冻结范围、checkpoint恢复及I/O/资源检查。聚焦测试 **33 passed**；新增执行器及两份相关测试的Windows编译、Ruff和`git diff --check`通过。本轮未重跑全套pytest；没有改变共享光学实现。attempt_01/02保留此前DLL搜索路径与过严逐位梯度一致性检查导致的工程失败；未将其计作科学结果，修复后attempt_03为有效完成结果。详见[报告](../outputs/20261009_m0_generator_energy_matched_target_comparison_v1/runs/attempt_03/report.json)、[输入人口](../outputs/20261009_m0_generator_energy_matched_target_comparison_v1/runs/attempt_03/input_population.json)、[A6700 checkpoint](../outputs/20261009_m0_generator_energy_matched_target_comparison_v1/runs/attempt_03/arm_a_step6700.pt)、[B6700 checkpoint](../outputs/20261009_m0_generator_energy_matched_target_comparison_v1/runs/attempt_03/arm_b_step6700.pt)及[训练图库](../outputs/20261009_m0_generator_energy_matched_target_comparison_v1/runs/attempt_03)。

## 2026-10-09 笔画评分收尾与呈现定位

评分入口已修复为读取实际A4700文件`readouts_start.pt`，A5400/B5400沿用各自读出，并从父报告读取`endpoint_metrics`。评分结果已落盘，另保留`report_awaiting_ratings.json`；相同答案可幂等恢复，不同答案拒绝静默覆盖。审阅质量为真实目标49/50、隐藏重复24/30（要求27/30），父任务终态为`REVIEW_INCONCLUSIVE / REVIEW_EVIDENCE_INSUFFICIENT`。起点/A/B人眼正确数为100/200、103/200、106/200；这组数不满足预设的人眼结论资格。

新增CPU复盘只消费已存图像与评分，没有读取母文件、重放光路或运行评价器。三端点500行的人口映射、标签、角色、目标和OLS一致；raw可逐像素重建生产28×28视图，600张主审图PNG与保存视图完全一致，30组隐藏重复PNG也逐像素一致。全1500行raw平均约0.199%像素超过1，截断强度约为raw总强度的0.277%（目标前景约0.376%）；生产顺序与“先池化后截断”的平均绝对差约0.000263，页面8位量化RMSE约0.00113。局部差异可以较大，但没有证据表明显示链错误或截断是类别2/4不可读的主因。类别2/4的前景raw MSE仍显著高于背景，分别约0.24–0.27与0.004–0.006；这说明输出形态与配对目标存在误差，不能单独证明具体笔画成因。

变更后的评分与定位聚焦测试 **15 passed**；变更文件Windows编译、Ruff及`git diff --check`通过。全过程模型更新、光学/评价器前向、梯度、采样和新增人工评分均为0；训练建议`NONE`、`next_action=NONE`。现阶段保留未知：显示链错配被排除，截断不是已证实主因，原始CMOS笔画形态与本次单人审阅一致性不足仍限制判断。

## 2026-10-08 冻结条件OLS生成入口

任务 `M0_GENERATOR_FROZEN_CONDITIONAL_OLS_GENERATION_V1` 固定训练人口扩展对照A4700 checkpoint，340,000个Generator相位参数、2048传播网格和严格FP32。采样器只用原200行canonical TRAIN（每类20行），以CPU float64拟合每类100维复OLS实部/虚部均值与总体标准差；固定seed `2026100813`，每类200张，先绘制全体实部噪声再绘制全体虚部噪声，生成OLS bank恢复时直接复用。没有相位修复、全局相位规约、重标定、拒绝筛选或S1采样。正式前向为200行经验OLS、10类均值OLS、2000行采样OLS共2210行；只读解码，不创建优化器，不计算梯度。

2000个采样结果的E_D正确数为1513/2000，宏召回75.65%，逐类为`196,192,170,199,199,66,151,156,109,75`；每类最低33%（类别5），类别9为37.5%。整体raw截断像素比例0.0846425%，单图最高截断0.8775%；所有输出有限、非零、非恒定。本次预设入口门槛（E_D宏召回≥50%、各类≥10%、整体raw截断≤1%及输出完整性）全部满足，故为`VALID_COMPLETED / CONDITIONAL_OLS_GENERATION_ENTRY_SUPPORTED`。E_C/E_I/E_D/E_S采样正确数分别为1859/2000、1741/2000、1513/2000、860/2000，E_S与其余评价器分歧明显。8位视图没有精确重复；最近样本与有限TRAIN目标距离只作描述，不用于新颖性或非记忆结论。无配对目标图像，不计算配对重建MSE。

同入口CUDA预检9项实际执行并通过，覆盖A4700加载、3300行TRAIN人口/角色、Encoder至缓存OLS重放、四评价器正控、类别均值和首样本生成解码、错误请求标签控制、模型/评价器/Adam冻结、写盘恢复及GPU/磁盘。正式前向后参数与buffer、评价器状态再次核对不变，所有梯度为空；训练/优化器更新0。GPU最初拒绝100行整块解码，修复为每次1行生产光路前向、仍按100行落盘；首次完整读取又暴露统计和图库起始块未对齐以及复数均值不能直接JSON序列化。保留无效尝试和故障记录，按明确manifest从已保存块恢复统计与报告，没有重抽样或重跑前向。最终attempt_02含23个读出块、20页2000样本图库、逐行float64评分、采样bank和终态报告；completed恢复检查不会追加工作。

同一执行中还完成了人口证据纠正：父训练覆盖attempt_03确有完整3300行三端点读出，但旧页只展示3000行，遗漏人口位置`[900,1200)`类别9历史确认集。CPU纠正输出见冻结任务attempt_02的`population_evidence_review.json`和`corrected_comparison_pages.json`：七个角色无重叠、无遗漏，补绘该300行的10页，修正清单按父`role_positions`、源行号、旧页序和30行分页尺寸映射历史页面。父报告和历史图未改写；因原报告未存逐页行号，纠正清单明确记录该重建依据。主判定仍只消费原覆盖对照的`[3000,3300)`人口，原`VALID_FAILED / MIXED_OR_INSUFFICIENT`没有被重判。两个入口的聚焦回归合计 **27 passed**；人口扩展与冻结入口变更文件编译、冻结入口Ruff及`git diff --check`通过；本轮未改共享光学代码，未重跑全套pytest。

## 2026-10-08 大 Generator 训练人口扩展覆盖对照

任务 `M0_GENERATOR_TRAINING_POPULATION_EXPANSION_COMPARISON_V1` 固定大Generator、原三评价器训练规则和A3700源状态，在相同1000次新增更新下比较A重复原200行（每行50次）与B覆盖2000行（每行5次）；每臂总逻辑图像呈现次数均为10,000。B人口由原200行和新增1800行组成。新增300行canonical TRAIN确认人口不参与训练、调参或中间选点。总测量人口3300行，含拟合、历史及旧确认角色；不能据此声称独立泛化或历史未见。

主判定人口E_D总正确数由A3700的217/300提高至A4700的230/300和B4700的231/300；B比A仅多1张，不满足预设的至少225张且高于A至少15张。逐类A4700为`29,26,27,24,30,15,20,25,19,15`，B4700为`27,25,27,26,30,17,22,28,18,11`。B使类别5、7分别比A增加2、3张，但类别9比A下降4张，未达到每类至少15张；类别0和1也各少2、1张。B宏召回77.00%，仅比A高0.33个百分点；B确认人口整体raw MSE为0.053929，低于A的0.055415及起点的0.056088，截断与图像完整性条件通过。总体科学分类为`VALID_FAILED / MIXED_OR_INSUFFICIENT`，`training_recommendation=NONE`、`next_action=NONE`。

这项同更新量对照检验的是覆盖更广与单图重复更少的组合效果，不能把干预归结为单独的数据集大小。结果没有支持“在本预算下扩大到2000行优于继续重复200行”；也不说明任何更大人口都无效。两臂均完成1000次Generator-only更新至Adam步数4700，200/2000行各自实际出现50/5次，批次映射和标签均衡核验通过；冻结范围、READY门槛和旧1200行源重放通过。三端点3300行读数齐全，但父attempt_03旧图页只覆盖3000行；其遗漏300行及角色映射已在后续冻结生成任务attempt_02中单独CPU纠正，详见本页“冻结条件OLS生成入口”。

执行中保留了两个无科学更新的工程失败尝试：attempt_01的临时恢复模拟缺少A完成标记，attempt_02的READY消费者错误地比较不同人口规模的整表指标；修复消费者并加入生产路径回归后，attempt_03完成有效测量。聚焦测试 **8 passed**；新增入口/测试的Windows编译、Ruff及`git diff --check`通过。本次未改共享光学路径，因此未重跑全套pytest。见[执行入口](../scripts/research/lightgen/execute_lightgen_2d_generator_training_population_expansion_comparison_v1.py)、[对应测试](../tests/research/lightgen/test_lightgen_2d_generator_training_population_expansion_comparison_v1.py)和[attempt_03输出目录](../outputs/20261008_m0_generator_training_population_expansion_comparison_v1/runs/attempt_03)。

## 2026-10-08 大 Generator 类别9加权 CE 对照

任务 `M0_GENERATOR_CLASS9_WEIGHTED_CE_COMPARISON_V1` 固定340,000参数大Generator和step3000状态，仅将三评价器真实类别CE中的类别9权重从1提高到2；A/B均继承相同Generator Adam及RNG，各追加700次Generator-only更新至step3700。只使用原200行canonical TRAIN更新，循环35轮；其余旧角色与新增300行确认人口均不进入梯度。新增人口按seed `2026100810`从排除既有900行后的canonical TRAIN抽取。总测量人口1200行：200拟合、100历史、300旧确认、300此前新确认、300本次新确认；不宣称独立泛化或历史未见。

主判定只看本次新300行。E_D正确总数为共同step3000的226/300（75.33%）、A3700的233/300（77.67%）和B3700的234/300（78.00%）。逐类数组分别为`29,26,30,27,28,10,21,22,21,12`、`29,28,30,27,28,11,21,24,21,14`、`29,28,30,26,28,11,21,23,21,17`。类别9达到17/30，比A多3张、比起点多5张；总正确数、类别1/7保护、单类退步上限和图像风险条件通过。唯一未通过条件是每类至少15/30，类别5在三个端点为10、11、11。故分类为`VALID_FAILED / MIXED_OR_INSUFFICIENT`：类别9加权有局部增益，但没有补齐十类覆盖；`training_recommendation=NONE`、`next_action=NONE`。

跨评价器看，新增确认人口类别9在E_S/E_I/E_C/E_D分别为起点`13/18/25/12`、A`14/18/25/14`、B`20/22/26/17`。四者真实目标类别9均通过正控；这说明B对该生成端点的类别9评分/分类读数一致上升，但不等同于人类可读性。B总raw MSE为0.057197，低于A的0.057324和起点0.057874；整体raw截断0.190%，类别1/7/9截断均低于0.2%，无零图或恒定图。固定色阶图仍显示模糊和散斑，需保留人工审阅，不以E_D计数替代可读性判断。

同入口READY预检12项全部实际通过：新300行四评价器目标正控、900行起点读数逐项重放、float32/float64加权CE解析梯度、A臂原规则参考、A/A重复、B臂3001—3003真实更新、错误标签负控、恢复一致性、冻结状态及源Adam不变。正式A/B各完成700步；检查checkpoint到3700，源模型与Adam起点逐项相同、Encoder冻结、Generator状态有限。训练记录抽核700条/臂，真实行号、标签、批次位置、权重策略和分母均一致；三端点各有完整1200行有限raw读数，起点与父读数在原容差内逐字段完全一致。固定色阶对照图覆盖全部五种人口角色。

实现及验证见[执行入口](../scripts/research/lightgen/execute_lightgen_2d_generator_class9_weighted_ce_comparison_v1.py)和[对应测试](../tests/research/lightgen/test_lightgen_2d_generator_class9_weighted_ce_comparison_v1.py)。聚焦测试 **10 passed**；Windows语法编译/模块加载、Ruff及`git diff --check`通过。本轮只增添独立执行器和测试，未修改共享光学实现，未重跑全套pytest。attempt_01保留输出原子硬链接被沙箱拒绝的工程失败；attempt_02保留NVRTC DLL搜索路径缺失的工程失败；attempt_03按受支持Windows DLL搜索路径和原子写入权限完成READY及训练。历史尝试均未覆盖，科学判定以[attempt_03报告](../outputs/20261008_m0_generator_class9_weighted_ce_comparison_v1/runs/attempt_03/report.json)为准。

## 2026-10-08 大 Generator 累计至3000步十类呈现

任务 `M0_GENERATOR_TEN_CLASS_PRESENTATION_ACCUMULATION_V1` 固定340,000参数大Generator、2048生产光路、Encoder/复OLS和原三评价器训练规则，从评价器消融A1700 checkpoint继承Adam状态，追加1300次Generator-only更新至3000步。更新只使用原200行拟合TRAIN，每行进入65次更新；100行历史比较、原300行确认和新增300行确认均不进入梯度。新增确认人口从排除旧600行后的canonical TRAIN中按seed `2026100809`固定抽取。900行测量角色为200拟合、100历史、300旧确认、300新确认；不作独立泛化声称。

新300行E_D正确数从A1700的203/300（宏召回67.67%）升至A3000的232/300（77.33%，净增29张）。逐类由`29,14,24,23,27,13,20,23,23,7`变为`30,27,26,27,28,15,21,25,22,11`；类别1和7保持或提升，类别5达到15/30，类别6为21/30，类别9为11/30。总宏召回提升、类别1/7不下降、单类损失上限及图像风险条件通过；唯一失败条件是十类各至少15/30，原因是类别9未达标。因此终态为 `VALID_FAILED / MIXED_OR_INSUFFICIENT`，`training_recommendation=NONE`、`next_action=NONE`。整体raw MSE从0.058352降至0.057162，raw截断比例为0.180%，无零图或恒定图；生成图像仍有明显模糊和散斑，分类器计数不等于人类可读性。

来源A1700读出逐项重放一致；训练记录共3000条，1701—3000的1300条每个20类均衡批次恰好65次，拟合位置和标签映射核对通过。CUDA READY实际检查覆盖四评价器目标正控、三步A/A更新、冻结组、负控、恢复一致性及READY消费者；所有必要检查通过。attempt_01—05保留了无科学训练的工程失败记录。attempt_06完成全部1300次更新并保存step3000；第一次收尾因历史基线读取边界失败，随后从显式step3000恢复完成两端900行读出。图像指标字段消费错误也根据已保存读数修复并回归；没有重复训练或光学重放。attempt_06的`invalid_harness.json`保留了初次收尾失败记录，最终有效测量与科学判定以`report.json`为准。

新增入口和测试分别见[十类呈现累计入口](../scripts/research/lightgen/execute_lightgen_2d_generator_ten_class_presentation_accumulation_v1.py)和[回归测试](../tests/research/lightgen/test_lightgen_2d_generator_ten_class_presentation_accumulation_v1.py)。聚焦测试 **11 passed**；变更文件Windows `compileall`、Ruff和`git diff --check`通过。没有修改共享光学或训练路径，未重跑全套测试。全部角色对照图、checkpoint、训练历史、900行端点读数及报告见[attempt_06输出目录](../outputs/20261008_m0_generator_ten_class_presentation_accumulation_v1/runs/attempt_06)。

## 2026-10-08 大 Generator 训练评价器消融

任务 `M0_GENERATOR_EVALUATOR_ABLATION_COMPARISON_V1` 固定340,000参数的大Generator、Encoder/复OLS、2048生产光路和原训练目标，仅比较A臂训练语义方向使用E_S/E_I/E_C与B臂使用E_I/E_C。两臂都从同一B1000模型、Adam状态和RNG独立副本起步，各追加700次Generator-only更新；200行canonical TRAIN拟合行每行使用35次，100行历史比较和300行确认行未进入梯度。B臂的E_S仅用于无梯度评分和记录，E_D两臂均只做前向。

来源B1000的600行重放逐项精确一致；同入口CUDA预检9项检查均实际执行并通过，包含A/A复现、两种梯度组合的独立参考、错误标签负控、三步真实更新、恢复一致性、冻结范围和源checkpoint不变。A/B起点权重及Adam逐项相同；两臂均完成新增700步，最终Adam步数1700，200个拟合行各进入35次更新。最终确认人口E_D宏召回为B1000 55.33%、A1700 67.33%、B1700 56.67%；类别1为5/30、20/30、13/30，类别7为14/30、18/30、10/30，类别8为18/30、20/30、27/30。B1700相对A恢复类别8并降低整体raw MSE（0.055369，对比A的0.057799和B1000的0.058463），但类别1/7没有同时超过两个基准，宏召回低于A，类别2、3、6、7仍有明显损失。科学终态为 `VALID_FAILED / NOT_SUPPORTED_WITHIN_700_UPDATES`；训练建议 `NONE`、`next_action=NONE`。这只否定本次固定目标、起点和700步下移除E_S的候选，不归因于E_S本身有害，也不否定大Generator的既有部分收益。

attempt_01因启动进程未包含受支持环境的CUDA DLL搜索目录而在NVRTC前向失败，未训练；attempt_02在预检中发现复用的 `L_dist` 与向量范数辅助函数来自错误模块，未训练。两次记录分别见[attempt_01](../outputs/20261008_m0_generator_evaluator_ablation_comparison_v1/runs/attempt_01/invalid_harness.json)与[attempt_02](../outputs/20261008_m0_generator_evaluator_ablation_comparison_v1/runs/attempt_02/invalid_harness.json)；attempt_03在修复并通过11项聚焦CPU测试后完成。新增执行器及测试的Windows `compileall`、Ruff和`git diff --check`通过；没有修改共享光学路径，因此未重跑全套pytest。结果、checkpoint、日志、读数及固定色阶图见[attempt_03输出目录](../outputs/20261008_m0_generator_evaluator_ablation_comparison_v1/runs/attempt_03)，实现见[消融执行器](../scripts/research/lightgen/execute_lightgen_2d_generator_evaluator_ablation_comparison_v1.py)和[回归测试](../tests/research/lightgen/test_lightgen_2d_generator_evaluator_ablation_comparison_v1.py)。

## 2026-10-07 大 Generator 累计呈现对照

任务 `M0_GENERATOR_CAPACITY_ACCUMULATION_COMPARISON_V1` 从大Generator快速对照的A/B各300步checkpoint继续训练，各保留本臂Adam状态并追加700步；模型结构、2048生产光路、目标函数和训练人口不变。200行canonical TRAIN拟合行进入更新，100行历史比较与300行确认行只用于指定读数；E_D只做前向。两臂均完成累计1000步，Encoder、OLS、物理缓冲与四个评价器保持冻结。

300步来源读数的重放逐张一致，最大差为0。确认人口E_D宏召回从共同起点0.3000升至A1000的0.3800和B1000的0.5533。类别1正确数分别为0/30、0/30、5/30；类别7为0/30、6/30、14/30。因此B1000的1/7都超过A1000和共同起点，宏召回也满足预设提升。但类别8从A1000的26/30、共同起点的23/30降至B1000的18/30，分别少8张和5张，违反最多减少3张的类别保护条件。

B1000确认人口整体raw MSE为0.0584632（A1000为0.0582866、共同起点为0.0593823），图像误差条件通过；整体raw截断比例为0.1181%，类别1和7分别为0.00475%和0.0667%。300至1000步期间，B臂E_D宏召回增加0.15，类别1从0升至5、类别7从7升至14，类别8仍为18；A臂宏召回增加0.0467、类别7从0升至6，类别1仍为0。完整结果为`VALID_FAILED / MIXED_OR_INSUFFICIENT`，训练建议`NONE`、`next_action=NONE`。固定色阶图仍显示明显的模糊/噪声纹理，不能把E_D计数提升称为人工可读性或完整语义生成。

首轮预检发现Adam加载共享步数张量、READY目录生命周期调用错误；正式A臂到1000步后的首轮测量又发现训练返回包装对象误传给端点测量。三项均作为`INVALID_HARNESS`记录保留；逐项修复后，attempt_03从明确A1000 checkpoint恢复并完成B1000训练、终点测量和判定。attempt_01和attempt_02未产生正式训练结果。聚焦回归 **12 passed**；变更文件Windows `compileall`、Ruff和`git diff --check`通过。未修改共享光学或训练路径，未重跑全套测试。入口及测试见[累计对照执行器](../scripts/research/lightgen/execute_lightgen_2d_generator_capacity_accumulation_comparison_v1.py)和[回归测试](../tests/research/lightgen/test_lightgen_2d_generator_capacity_accumulation_comparison_v1.py)。

## 2026-10-07 大 Generator 快速对照

任务 `M0_GENERATOR_LARGE_CAPACITY_QUICK_COMPARISON_V1` 固定step2500 B端点、Encoder/OLS和2048生产光路，对比20,000参数的小Generator与340,000参数（17倍）的大Generator。两臂各仅更新Generator 300步，使用相同200行拟合批次；100行历史比较与300行确认人口不进入梯度。大Generator新增两张400×400相位面，置于原1.5 mm两段之间；零相位扩容在600行上的raw CMOS与OLS输出均精确一致。大模型训练时每次仍以10行为逻辑批，内部采用2行微批；通过完整批次OLS统计导数的链式累计保留了原BOGT `L_dist` 定义。独立CPU全批反传对照最大梯度误差4.44×10⁻¹⁶，中心有限差分误差3.89×10⁻⁹；B臂同入口三步CUDA预检通过。

最终报告为 `VALID_FAILED / MIXED_OR_INSUFFICIENT`。确认人口E_D宏召回由B起点0.3000升至B终点0.4033，较A终点0.3333高0.0700；类别7正确数由0/30升至7/30，类别1仍为0/30。类别8从A终点24/30降至18/30，违反每类最多减少3张的护栏；类别1/7联合增益条件也未满足。B终点整体raw MSE为0.0591188（A为0.0590596，B起点为0.0593823），整体截断比例0.0856%，图像风险护栏通过。故本次大Generator结果显示有限预算下的部分收益与类别取舍，既不支持当前扩容候选，也不证明所有大架构无效或小架构容量不足。

预检_09的全程显存峰值记录超过90%安全线，未用于正式训练；采用精确逻辑批微分解后，预检_10 READY。正式attempt_03全程1,903次采样峰值4,142/6,144 MiB；[显存峰值汇总](../outputs/20261007_m0_generator_large_capacity_quick_comparison_v1/runs/attempt_03/resource_peak_observation.json)、逐点样本、三步预检与A/B完整checkpoint均在attempt目录。第一次finalizer因配对统计读取`ed_logits`而失败；改为生产读出键`e_d_logits`后，新增测试并通过显式B300恢复完成最终报告，没有增加训练或光学重放。`attempt_01/02`的工程失败记录以及attempt_03内的原始`invalid_harness.json`均保留。新增入口和测试分别为[大Generator对照入口](../scripts/research/lightgen/execute_lightgen_2d_generator_large_capacity_quick_comparison_v1.py)和[回归测试](../tests/research/lightgen/test_lightgen_2d_generator_large_capacity_quick_comparison_v1.py)；聚焦测试 **10 passed**，新增文件Windows `compileall`、Ruff及`git diff --check`通过。没有修改共享光学路径，未重跑全套pytest。

## 2026-10-06 类别保护共同方向冻结验证

任务 `M0_GENERATOR_CLASS_GUARDED_SHARED_DIRECTION_V1` 只在冻结的B端点上计算30个三评价器类别CE梯度并构造一个共同投影位移，没有模型或优化器更新。有效报告位于[attempt_03](../outputs/20261006_m0_generator_class_guarded_shared_direction_v1/runs/attempt_03/report.json)：30项梯度与原混合批次CE梯度的最大相对L2误差为2.10×10⁻⁷；投影KKT最大残差4.78×10⁻⁹（门槛1×10⁻⁸），保留原纯CE方向L2范数的84.2%。同入口CUDA预检为READY；四评价器目标正控通过，分数差相关中心差分、冻结状态、写盘恢复及完成态恢复核验均通过；终态恢复未追加文件或测量。

确认人口的候选正向位移令类别1/7分数差分别有29/30和25/30行改善，宏召回仍为0.30、两类正确数仍各0/30。30项训练评价器平均CE保护均未超过1×10⁻⁴容差；但类别9有24/30行稳定有害，类别8和9平均分数差下降超过门槛，因此终态为 `CLASS_GUARDED_RESPONSE_WITH_TRADEOFF`。确认集raw MSE仅从0.0593823变至0.0593874，截断比例约0.05%，零图和恒定图均为0。结论限于一个冻结共同位移的局部响应，不能说明累计训练会形成正确数字，也不构成架构容量证据。attempt_01因过严的梯度一致性数值比较、attempt_02因自测manifest语义字段不完整而以 `INVALID_HARNESS` 结束；两者均为零更新工程记录，不影响attempt_03有效结果。

## 2026-10-07 CE与分数差联合保护数值预检

任务 `M0_GENERATOR_CE_MARGIN_GUARDED_SHARED_DIRECTION_V1` 继续只做冻结方向构造和零更新数值预检，没有创建优化器、更新模型或测量600行人口。输入人口、固定端点、父CE梯度来源、60项投影约束及KKT核验通过；KKT最大残差9.88×10⁻⁹。入口和测试升级到v2数值证据格式，评分生产函数在保留光学/评价器float32前向的同时，以float64逐行计算CE和分数差；有限差分采用逐行正负配对后再取float64均值，最终评分与预检共用相同定义。测试覆盖生产logits到最终方向判定，而不只检查手填斜率。

[attempt_06数值修复](../outputs/20261006_m0_generator_ce_margin_guarded_shared_direction_v1/runs/attempt_06_numerical_repair/failure.json)保存了48项检查和逐行诊断，但因Torch版本元数据未转成安全序列化类型，证据复读失败，故该attempt仍为 `INVALID_HARNESS`。attempt_07修正了证据序列化并可用安全加载器复读；float64逐行配对归约没有解释原失败：48项中35通过、13失败，与旧float32归约斜率差最多约1.31×10⁻⁶，而失败误差最大约5.88×10⁻⁴。no-grad和autograd的raw CMOS、评价器视图与logits一致，独立相位/CMOS/视图梯度链路参考误差约10⁻⁷。最大错误类别没有切换，CMOS clamp分支没有变化；有限扰动中可见评价器ReLU与MaxPool分支改变，可能参与有限尺度响应偏离局部导数，但还不能确认为全部失败项的唯一原因。

额外严格FP32（关闭TF32）诊断的48项有限差分全部通过；但它在完整600行、六个100行块的基线重放中均无法匹配保存的E_S logits，因而不兼容原测量基线。流程按计划停在基线兼容性检查，未把严格模式结果视作READY；严格模式下的CE来源一致性和新候选完整检查均未执行。attempt_07的[失败终态](../outputs/20261006_m0_generator_ce_margin_guarded_shared_direction_v1/runs/attempt_07_strict_fp32_compatibility/failure.json)与[数值诊断](../outputs/20261006_m0_generator_ce_margin_guarded_shared_direction_v1/runs/attempt_07_strict_fp32_compatibility/preflight_diagnostics.json)记录全部证据。原48项未通过，故没有READY、正式基线/正负位移测量或科学分类；该工程失败不能重判父任务，也不构成候选方向有害、目标失败或容量不足的科学结论。

聚焦测试 **12 passed**；新增入口和测试的Windows `compileall`、Ruff及`git diff --check`通过。没有运行全套测试，因为未修改共享路径；没有模型或优化器更新。联合保护假设仍未获有效检验，最近有效科学结果仍是类别保护方向的 `CLASS_GUARDED_RESPONSE_WITH_TRADEOFF`。`training_recommendation=NONE`、`next_action=NONE`；本次停止，不启动累计训练或扩容。

## 2026-10-06 代码审查修复

本轮仅修复恢复输入核验、重建任务输出目录保护和 Windows 证据长路径读写。BOGT 新 run 保存 `resume_inputs.pt`，生产恢复入口在构造训练消费者前逐项比较实际图像、标签、TRAIN/DEV 行号及评估器状态；内容、形状或 dtype 不同即作为工程错误停止。摘要仍仅作溯源；同一输入重新序列化不会因摘要变化而拒绝恢复。缺少实际输入快照的旧 run 无法证明输入未变，明确拒绝恢复，不从当前文件补造历史基线，也不修改旧产物。完整 MNIST 快照增加约180 MiB/run的存储开销。

Generator-only 重建入口只接受 `outputs/YYYYMMDD_<version>/runs/<run>`，在创建目录前拒绝 `final_pt`、恢复元数据和非 run 路径。共享证据 IO 使用 Windows 扩展路径及短临时文件名，保持完整写盘后原子发布、拒绝覆盖既有新产物的语义；Torch 写盘通过文件句柄，加载器和溯源读取器也支持长路径。回归覆盖实际输入生产/恢复、生产入口拒绝决策、受保护路径及四类原子写盘的长路径加载和重复发布。

最终版本聚焦回归 **135 passed**；本轮变更 Python 文件的 Ruff `E4/E9/F821`、外部字节码目录 `compileall` 和 `git diff --check` 通过。完整默认测试为 **2087 passed, 8 failed, 7 errors, 13 deselected, 25 subtests passed**（508.94秒）；失败涉及历史 authority 文件/目录权限拒绝及既有 migration envelope 检索为空，因此不宣称全套通过。代表文件普通路径与扩展路径均返回权限拒绝，未修改科研文件权限。全套在长路径读取兼容补充前加载了共享 IO；补充后受影响的135项聚焦检查已重跑通过，尚不宣称最终版本的全套绿灯。本轮没有启动科学训练、CUDA预检或测量，没有更新科学结论；没有提交、推送或刷新 Sources。

## LightGen 主线

当前问题是 BOGT 联合目标能否经统一 Encoder→复 OLS→Generator 映射产生可辨识的 CMOS 数字图像。规范梯度曾改善 DEV 和 TRAIN-neighbor S1 的独立 E_D 读数，但 S1 未达到原生成门槛。small-bank step2200 是研究证据；其原训练 attempt 仍为 `INVALID_HARNESS`，只读对账没有追认 Gate。S1 属于 TRAIN 邻域采样，不代表独立先验新颖性。

最近的类别1/7定位任务 `M0_BOGT_CLASS17_JOINT_MAPPING_ATTRIBUTION_V1` 保持 `attempt_01=INVALID_HARNESS`、参数更新为0；固定 step1200/step2200 前向读数已保存，但探针、梯度和完整报告未完成。失败的 `preflight_04` 没有启动科学测量，因此类别1/7瓶颈仍无定位结论。已停止 M0 执行器归档，不是当前可执行入口。历史见[整理前状态快照](archive/20261001_project_refactor/CURRENT_STATUS_pre_refactor.md)及[实验历史](status/experiment_history.md)。

随后完成了独立任务 `M0_CLASS17_MAPPING_DIAGNOSTIC_V2`，结果为 `VALID_COMPLETED`，见[冻结诊断报告](../outputs/20261002_m0_class17_mapping_diagnostic_v2/runs/attempt_01/report.json)。它固定 step1200/2200 和 300 行 canonical TRAIN（200 拟合、100 观察），没有光学参数或 OLS 更新。step2200 的去相位 OLS、未截断 CMOS 池化、评价器视图探针均支持类别1/7；生成图像 E_D 不支持这两类，而目标图像 E_D 正控宏召回为 0.993、类别1/7召回均为1.0。两类评价器视图前景 MSE 高于背景 MSE，raw 截断比例均低于1%，因此唯一建议是待检验的固定 Encoder/OLS、Generator-only 前景/背景等权重重建对照。该候选不代表完整语义生成成功或已证明因果；报告停止点为 `next_action=NONE`，没有后继训练授权。报告所列 6.61 秒是已保存证据的最终恢复汇总耗时；原测量阶段遥测最大记录为61.52秒，两者都只作耗时观察。

按该建议完成的[Generator-only 前景/背景重建对照](../outputs/20261002_m0_generator_balanced_reconstruction_v1/runs/attempt_02/report.json)为 `VALID_COMPLETED`：固定同一step2200起点、300行TRAIN（200更新、100观察），A/B各更新1000步，仅Generator相位可更新。目标图像E_D正控观察宏召回0.99，类别1/7各10/10。观察集step2200/A/B宏召回为0.34/0.35/0.34，类别1/7三端点均为0/10；B的原始前景MSE较A下降约5.0%（1类）和3.1%（7类），未达到预设两类各下降10%的门槛，Holm校正McNemar均为p=1。整体原始MSE降低且截断低于1%，但语义、宏召回、类别保持及前景误差联合标准未满足，最终为 `MIXED_OR_INSUFFICIENT`，训练建议 `NONE`。这削弱了“仅改Generator重建权重即可解决1/7语义呈现”的假设；不构成架构容量不足证据，也不授权扩大架构或继续训练。聚焦测试 **14 passed**；全套 pytest **1956 passed, 1 failed, 13 deselected, 25 subtests passed**。唯一失败是独立 `near_conjugate` 诊断测试的输出父目录缺失，单测复跑可复现，与本次LightGen任务无调用关系；不在本次修改范围。Windows `compileall`、本轮及相邻改动文件 Ruff、`git diff --check` 通过。预检和恢复中的工程失败记录均保留在attempt目录，最终有效报告记录了恢复经过。

随后完成[已有400模型冻结重放](../outputs/20261002_m0_existing_400_frozen_replay_validity_v1/runs/attempt_01/report.json)，任务 `M0_EXISTING_400_FROZEN_REPLAY_VALIDITY_V1` 终态为 `VALID_COMPLETED / FROZEN_REPLAY_CONFIRMED`。step600模型按原400×400、2层Encoder→100维复OLS→4层Generator、512传播网格代理配置只读重放，训练更新为0；300行canonical TRAIN只作参考。2000张28×28视图和映射的50张raw图均在 `rtol=1e-5, atol=1e-6` 下匹配历史产物；历史/新E_D混淆矩阵与原报告一致，预测翻转为0。E_D生成宏召回0.8545、类别1为0.485、类别7为0.93，raw超过1的比例0.8579%，目标图像正控宏召回0.9933且类别1/7均为1.0。复现确认了已有数值读数，但类别1的生成呈现风险仍在，不能据此声称完整语义生成、证明训练机制或判断需要大架构；训练建议为 `NONE`、`next_action=NONE`。全类图、raw对照和300行有限TRAIN近邻图在attempt目录。CUDA预检与分块恢复中两项工程故障已修复并保留记录；最终显式恢复核验未追加前向。聚焦测试 **15 passed**；Windows `compileall`、Ruff、`git diff --check` 通过。全套 pytest **1992 passed, 1 failed, 13 deselected, 25 subtests passed**；唯一失败是既有无关的 near-conjugate 诊断写盘缺失父目录问题，本次未修改。

随后完成[原生复OLS依赖与呈现冻结诊断](../outputs/20261004_m0_native_ols_dependence_and_presentation_diagnostic_v1/runs/attempt_01/report.json)，任务 `M0_NATIVE_OLS_DEPENDENCE_AND_PRESENTATION_DIAGNOSTIC_V1` 终态为 `VALID_COMPLETED / MIXED_OR_INSUFFICIENT_BASELINE_FLOOR`。没有创建优化器或更新模型；固定300行canonical TRAIN分为200行拟合角色与100行观察角色。step2200主线走真实2048网格图像→Encoder→复OLS→W_OLS→Generator，400模型保留原400×400、512网格代理光路，且只重用既有冻结重放确认过的2000行生成读数。E_D目标正控宏召回0.993，类别1/7均为1.0。主线观察角色的原始读数中，E_D类别1/7均为0/10，E_C类别1为9/10、类别7为0/10，因此主线不满足置换归因的基线支持条件；类别1/7观察结果不能用于定位跨端口OLS依赖。400代理原始E_D类别1/7为7/10与9/10，E_C为10/10与8/10；功率匹配的端口置换呈现不一致变化，首个种子的四项E_D配对检验Holm校正后p值为0.578、1.0，未达到预设门槛。两条路径的共享整行负控均保持五评价器混淆矩阵不变；按真实生成器入口平面功率匹配后的最大相对误差为1.2×10⁻⁷。28×28视图平移/模糊只作分类器敏感性读数，不代表光学或校准改变。该任务未证明OLS依赖是主线瓶颈，也未改变“完整语义生成尚未达标”的主线结论；不支持扩容或启动下一训练，建议为 `NONE`、`next_action=NONE`。

新入口为[冻结诊断脚本](../scripts/research/lightgen/diagnose_lightgen_2d_native_ols_dependence_v1.py)，对应[回归测试](../tests/research/lightgen/test_lightgen_2d_native_ols_dependence_v1.py)。聚焦测试 **8 passed**；全套 pytest **2045 passed, 13 deselected, 25 subtests passed**；Windows `compileall`、本次入口与测试文件 Ruff、`git diff --check` 通过。attempt目录保存了预检、行级分块、两种OLS置换种子、功率匹配系数、评价器读数、呈现对照图与最终报告。显式终态恢复检查返回 `additional_forward=false`。

随后执行了[Generator-only 三评价器语义目标对照](../outputs/20261004_m0_generator_semantic_target_comparison_v1/runs/attempt_01/report.json)：固定step2200起点，A为实例特征距离、B为三评价器真实类别CE，Generator-only各更新300步；原200行TRAIN用于更新、旧100行仅作历史比较，另以300行未参与更新的TRAIN作确认。四个评价器的目标图像正控均通过。确认集E_D宏召回由step2200的0.270升至A的0.283、B的0.300，但B相对A仅+0.0167，低于预设+0.05；类别1和7在起点、A、B均为0/30，配对McNemar的Holm校正p均为1.0。B的确认集raw MSE、截断和数值有效性条件通过，其他类别也未超过预设退步限值；然而类别1/7提升与宏召回门槛均未满足，终态为 `VALID_FAILED / NOT_SUPPORTED_WITHIN_SCOPE`，训练建议 `NONE`、`next_action=NONE`。本结果否定的是“在本次固定起点、人口和300步预算下，该CE目标已带来可转移的1/7收益”，不判断架构容量，也不改变完整语义生成尚未达标的主线结论。完整数值、逐类混淆矩阵、三端点读数、图像统计与checkpoint均保存在attempt目录；[任务入口](../scripts/research/lightgen/execute_lightgen_2d_generator_semantic_target_comparison_v1.py)和[对应测试](../tests/research/lightgen/test_lightgen_2d_generator_semantic_target_comparison_v1.py)。预检早期两条`INVALID_HARNESS`分别因目标张量设备不匹配和NVRTC DLL搜索路径问题；修复后`preflight_recheck02`为READY，均为工程记录而非科学结果。原始遥测保留在旧writer命名文件，[工程来源说明](../outputs/20261004_m0_generator_semantic_target_comparison_v1/runs/attempt_01/engineering_provenance.json)记录该标签错配及逐行来源，报告所指向的任务名遥测为其无损伴随记录。聚焦测试 **9 passed**；Windows `compileall`与新增入口/测试 Ruff通过；本轮全套pytest **2054 passed, 13 deselected, 25 subtests passed**（仓库外basetemp）；`git diff --check`通过，新增状态页链接均核对存在。

随后完成了[类别CE训练的零更新证据复盘](../outputs/20261004_m0_semantic_target_evidence_review_v1/runs/attempt_03/report.json)，终态 `EVIDENCE_REVIEW_COMPLETED`；父训练结论仍是 `VALID_FAILED / NOT_SUPPORTED_WITHIN_SCOPE`。复盘仅使用已保存读数和日志，没有模型/光学/评价器前向、更新、梯度、探针拟合或重采样。三端点600行的人口、标签和角色核对通过；冻结目标及OLS跨端点最大差为0；从logits重算的分类计数、混淆矩阵与父报告一致，原配对检验和Holm值也复现。训练记录确认A/B每个拟合行均更新15次、标签符合批次；原`source_row_indices`字段实际记录0—199的人口位置，复盘保留原日志并提供到真实母文件行号的映射表。B臂首末固定训练周期的三评价器CE监测值均下降，但E_D确认宏召回仅从起点0.270、A的0.283变为B的0.300；类别1和7的E_D正确数三端点仍均为0/30。B对类别7的E_I/E_C计数相对A有所上升，类别1的E_C正确数从27/30降至21/30，显示评价器响应不一致。现有证据不能区分评价器梯度冲突、继承Adam方向和光学图像呈现问题；不作架构容量归因，不建议训练改动，`training_recommendation=NONE`、`next_action=NONE`。报告列出唯一待设计测量问题，未授权实施。行级分数/转换、逐更新和同批次首末轮汇总、修正行号映射及10页五类审阅图均在attempt目录；[复盘入口](../scripts/research/lightgen/review_lightgen_2d_semantic_target_evidence_v1.py)与[回归测试](../tests/research/lightgen/test_lightgen_2d_semantic_target_evidence_review_v1.py)。聚焦测试 **9 passed**，编译和Ruff通过；未重跑全套测试。`attempt_01`保留序列化中断记录，`attempt_02`保存首次完成复算的报告，`attempt_03`再加入逐批次首末轮对照表作为最终复盘版本；父实验与原训练日志均未覆盖。

随后完成[Generator语义方向与Adam作用冻结诊断](../outputs/20261004_m0_generator_semantic_direction_adam_diagnostic_v1/runs/attempt_02/report.json)，任务 `M0_GENERATOR_SEMANTIC_DIRECTION_ADAM_DIAGNOSTIC_V1` 为 `VALID_COMPLETED / MIXED_OR_INSUFFICIENT`。诊断冻结step2200、A终点和B终点，只对原200行拟合TRAIN执行完整2048光路前向与只读梯度计算，光学参数/Adam更新为0；旧100行与新增300行没有重新打开。目标正控的E_S/E_I/E_C宏召回均为1.0，E_D为0.995，测量资格通过；60个拟合批次的保存读出均通过父读数重放容差，报告记录预测翻转为0。E_D参与了只读方向诊断，后续确认不能称为E_D盲测。B端点类别7的语义CE下降方向有16/20行一阶有利，但加入BOGT后的下降方向为15/20，尚未达到16行稳定线；类别1语义方向为12/20有利、7行边界，完整合成方向仅9/20有利。保存Adam、零一阶矩、完整清零及BOGT投影Adam均未对两类形成预声明的稳定有利模式，因此不支持重置Adam或投影辅助梯度。结论仅描述局部方向，不意味着越过E_D分类边界或形成可读数字，也不支持扩容或续训；建议 `NONE`、`next_action=NONE`。完整逐行/逐批次候选方向、三个端点的读出块和恢复信息见报告目录；[诊断入口](../scripts/research/lightgen/diagnose_lightgen_2d_generator_semantic_direction_adam_v1.py)、[测试](../tests/research/lightgen/test_lightgen_2d_generator_semantic_direction_adam_v1.py)及[工程过程补充](../outputs/20261004_m0_generator_semantic_direction_adam_diagnostic_v1/runs/attempt_02/engineering_provenance.json)。attempt_01和attempt_02中的失败文件均保留；补充说明更正attempt_01失败文本对“CUDA测量尚未开始”的宽泛表述，实际READY预检已完成但正式测量未开始。最终恢复只汇总既有60块，没有追加前向或梯度。聚焦测试 **12 passed**；最终全套 pytest **2075 passed, 13 deselected, 25 subtests passed**；Windows `compileall`、新增入口与测试 Ruff、`git diff --check` 均通过。

**2026-10-04方向证据纠正：** 本轮源码复核及[CPU解析反例](../outputs/20261004_m0_generator_semantic_direction_adam_diagnostic_v1/runs/diagnostic-only_margin_gradient_code_review_01/margin_gradient_code_review.json)确认上述方向诊断存在实现缺陷：入口在计算最大错误类别logit时先执行`detach()`，因此报告中的分数差数值仍正确，但其反向导数只含真实类别logit项，遗漏最大错误类别项。生产表达式在解析例中把正确导数−1算成+1；所有logit共同平移时，真实分数差导数为0，生产表达式却得到+1。故attempt_02的点积、夹角、16/20模式与最终方向分类暂不采信，属于方向测量的`INVALID_HARNESS`工程问题；原报告保留，不据此产生新的科学负结论。此前12项聚焦及全套测试没有覆盖生产分数差的完整导数定义。此外，生产调用未传递最大错误类别并列标志；当前保存的300个关注/比较行没有并列，未影响本次计数，但恢复验收前需补齐该出口。光学/OLS/logits前向证据、标量分数差、父CE训练的`VALID_FAILED / NOT_SUPPORTED_WITHIN_SCOPE`及零更新复盘计数均保留。下一项计划应先修复可导分数差与并列出口，在相同端点、200行拟合TRAIN和原阈值下冻结重测；尚未实施修复或启动CUDA，训练建议仍为`NONE`。

## 工程扫尾

本轮完成 LightGen 主线及其直接依赖的小范围工程整理，覆盖现存 BOGT 训练、三评价器规范方向、S1 采样和相关加载/恢复路径。恢复不再以源码、文件摘要、dirty 状态或 Git HEAD 相等作为准入条件；实际加载的模型/配置、更新步数与 Adam、样本行与标签、数据角色、冻结范围及数值状态仍会校验。耗时保留为观测，不作为硬拒绝；硬件安全检查保留。

三评价器规范方向的共享计算位于 `src/ldm_project/v2/application/research/lightgen/canonical_semantics.py`，BOGT 训练消费位于 `src/ldm_project/v2/application/research/lightgen/lightgen_2d_chen_bogt_complex_ols_training.py`。目录测试现在检查 Git 中实际存在的配置；pytest 默认不写仓库父目录缓存，测试临时目录使用外部可写路径。停止任务和无当前消费者的研究家族没有在本轮扩展整理。

验证结果：聚焦 pytest **101 passed**；全套 pytest **1932 passed, 13 deselected, 25 subtests passed**；Windows `compileall`、`git diff --check` 通过。Ruff 对本轮变更通过；三评价器历史脚本已有的 E702 分号格式问题单独排除，未扩展成历史格式清理。

类别1/7诊断入口为[诊断脚本](../scripts/research/lightgen/diagnose_lightgen_2d_class17_mapping_v2.py)，并有对应测试；恢复身份比较保留实际模型配置检查，并忽略 Git 容器与迁移后的 `runtime_module` 路径差异。该 attempt 的无更新 CUDA 预检为 `READY`，证据写入默认 attempt 目录；报告终态有效。诊断聚焦测试 **34 passed**；最终全套 pytest **1943 passed, 13 deselected, 25 subtests passed**；Windows `compileall`、改动文件 Ruff、`git diff --check` 和状态页相对链接检查通过。全套验证使用仓库外短路径 basetemp。

### 2026-10-02 工程治理整改

共享研究生命周期现在要求完整 READY 凭据；三个检查组各自带有生产者计算的 `passed` 结果，旧的无凭据启动方法会拒绝启动。BOGT `--execute` 会在进入科学 attempt 前检查显式加载状态，在实际 2048 网格和同一训练消费者路径上执行三步隔离更新、冻结参数负控及 checkpoint/Adam/RNG/采样器回滚验证；通过后恢复原状态并把凭据引用写入报告。新 attempt 若生产预检或开闸失败，会删除该次尚未打开的 attempt 目录并保留独立工程失败记录；恢复既有 attempt 时保留原产物。架构边界测试改为扫描真实 v1/v2 实现，README、导航、环境说明和 AGENTS 合同已同步。

上一轮默认全套 pytest **1969 passed, 13 deselected, 25 subtests passed**，后续聚焦回归 **106 passed**。2026-10-03 复核撤掉了无依据的固定空闲显存和占用率门槛；显存和耗时只作观察，真实 CUDA/分配失败按工程错误处理。BOGT 的 CPU 与 CUDA 预检共用实际三步更新、checkpoint 重载与回滚逻辑，从已加载的模型和 Adam 状态开始，并核对步数、moment 数值、采样器及历史。冻结违规和准备阶段 OOM 也由回滚/失败记录测试覆盖。复核还修正了多个凭据生产器的字段错配和空回滚结果被误判为通过的问题；架构测试扩大到真实 v1/v2 与 v2 current 实现。

上一轮核心聚焦回归 **84 passed**，随后联合执行器/近恒等入口凭据回归 **30 passed**；最终默认全套 pytest **2006 passed, 13 deselected, 25 subtests passed**（390.55秒）。Ruff `E4/E9/F821`、Windows `compileall`、当时的文档链接核验和 `git diff --check` 通过；该 Ruff 门禁不宣称旧研究脚本的完整格式基线干净。

### 2026-10-03 内部工程验收收尾

共享凭据别名冲突检查、生产预检更新步数/采样器/历史/Adam一致性、checkpoint中途重载和成功/失败/冻结违规回滚已补行为测试。新增哈希锁校验器和共享 CPU Actions 执行脚本；Windows CPU Actions 与本地使用同一34包 SHA-256 锁和同一检查命令。

完整默认 pytest 在短路径 `C:\TEMP\pt` 下通过：**2026 passed, 13 deselected, 25 subtests passed in 391.89s**。独立34包 CPU 锁环境的安装、`pip check`、editable安装与CPU张量 smoke通过；首次完整Windows CUDA锁安装则复现了原环境已有的依赖冲突。该冲突已由本节最新环境锁修复记录闭环。

BOGT `--production-preflight` 在实际 CUDA 生产路径分别读取原 `attempt_01` 的显式 step0 和 step100 checkpoint，使用2048传播网格、55,000行TRAIN。两次均为 `READY`，各做3个隔离更新；冻结负控拒绝、checkpoint中途重载对照和十项完整恢复验证通过。step0恢复到0步/0历史/空Adam/采样游标0，隔离更新1–3；step100载入100步/100行历史/2组Adam状态/游标100，隔离更新101–103，第一批由载入采样状态的下一批组成。两次均未开科学attempt、未授予科学权限、科学更新为0。报告记录首批TRAIN行号、模型/冻结状态/参考/权重/Adam/梯度/RNG/采样器/进度恢复检查和资源观察：[step0报告](../outputs/20261003_engineering_governance/runs/diagnostic-only_bogt_preflight_step000000_pathfix_v4/preflight_report.json)、[step100报告](../outputs/20261003_engineering_governance/runs/diagnostic-only_bogt_preflight_step000100_pathfix_v1/preflight_report.json)。

NVRTC首轮启动错误不是文件缺失：CUDA runtime与`torch/lib`目录内的DLL存在，但原启动进程搜索路径未包含它们。仅在验收进程前置受支持环境的`bin`、`Library/bin`、`Scripts`和`torch/lib`后，真实CUDA kernel与生产预检通过；未改受支持环境或系统设置。观测到的空闲显存、保留显存和耗时仅记录，不设准入门槛。

首次完整Windows锁安装按原169项Conda包和2项pip包成功重建，但`pip check`复现了受支持环境的两项Torch依赖问题：缺少`fsspec`、SymPy版本不符。首次失败环境与日志保留为历史证据，原锁快照见[修复前锁文件](../outputs/20261003_engineering_governance/runs/diagnostic-only_windows_lock_repair_20261003/pre_repair_snapshot/environment-win-explicit.lock.txt)。

### 2026-10-03 Windows锁修复复验

最小修复保留Torch `2.5.1`、torchvision `0.20.1`和CUDA `12.1`：Conda锁将SymPy固定为`1.13.1`及其兼容的mpmath `1.3.0`；pip哈希锁改为fsspec `2026.7.0`和PyMuPDF `1.28.2`。Shiboken6由Conda PySide6文件记录提供，安装核验器会检查该归属。渠道校验现在接受锁定源的合法`noarch`／平台子目录，并拒绝错误渠道、主机和路径。

新的Windows前缀`C:\TEMP\mmr5\env`按169项Conda锁和两个带哈希pip包创建。Conda版本/构建/URL/SHA-256/渠道记录169项全部通过；SymPy和mpmath归档SHA-256匹配；pip hashes、Shiboken归属、editable安装、`pip check`、项目导入及CPU张量检查均通过。最终运行信息为Python 3.11.16、Torch 2.5.1、torchvision 0.20.1、CUDA构建12.1；CUDA可用性只记录为观察值。本轮更新后的Windows Actions等价本地脚本为**164 passed**，Ruff `E4/E9/F821`和compileall通过。详细过程、历史失败前缀、旧锁快照及检查日志见[Windows锁修复验收记录](../outputs/20261003_engineering_governance/runs/diagnostic-only_windows_lock_repair_20261003/acceptance_record.json)。

| 方面 | 本轮已落实 | 剩余范围或验证限制 |
| --- | --- | --- |
| 文档管理 | `CURRENT_STATUS.md` 汇总动态结论；操作指南提供锁安装、共享CPU检查和生产诊断入口；handoff保留输入与停止点；导航增加生产诊断与400模型冻结重放。 | 历史归档按当时事实保留，不作当前状态来源。 |
| 代码管理 | BOGT继续复用生产训练消费者、隔离更新、checkpoint重载和回滚；`--production-preflight` 独立诊断后退出；Actions和本地检查走同一脚本。 | 其他大型/历史入口未全面拆分或迁移；本轮按实际边界修复，不按行数重写。 |
| harness设计 | READY要求正确凭据、三个通过检查组及一致实现选择；兼容别名冲突被拒绝；终态重启仍被拒绝。 | 历史任务常量、目录位置和诊断凭据均不授权后续科学任务。 |
| harness运用 | step0/100真实2048 CUDA三步更新、负控、中途checkpoint重载、步数/Adam/历史/采样器一致性及完整回滚均通过；完整默认套件通过。 | 不包含终段检查点三步窗口；没有启动研究训练、DEV测量或生成。 |
| 防御性编程 | 输入、加载状态、冻结范围、别名语义、负控、数值有效性和异常后全状态恢复由生产路径和行为测试验证。 | 仍未逐项审计所有历史脚本的元数据内部语义。 |
| hash与Git/环境 | SHA-256 Conda/pip/CPU锁、安装记录校验、渠道格式回归、Windows锁干净重建与CPU本地workflow通过；Actions等价脚本164 passed，Ruff和compileall通过。 | 远端CI因不推送未触发，继续待验收；本地锁安装、依赖闭环和CPU smoke已通过。 |
| AGENTS.md | 根与scoped合同保持权限/作用域路由；锁一致性、导入边界、凭据与恢复承诺有机器检查。 | 自然语言权限和全部旧脚本规则并未全部形式化。 |

### 2026-10-03 C 盘临时资产清理

以下记录的是后续迁移前的清理阶段状态；当时明确保留的两套 C 盘验收环境，已在下面的 D 盘迁移中通过验收后删除。

按明确路径白名单清理本轮遗留的临时环境、重复缓存和测试目录；目标均位于 `C:\TEMP`，盘点和复核未发现重解析点。23 个目标现在全部不存在：22 个在清理过程中删除，另一个空目录在扩展路径删除探测时已删除。保留 Windows 锁环境 `C:\TEMP\mmr5\env`、CPU 锁环境 `C:\TEMP\codex-governance-cpu-locked-env-2dd386bf923444ba8da546316f5086b9\venv`、micromamba 工具和私有运行根目录、原支持环境及 D 盘证据。C 盘可用空间在清理前为 **54.4701 GiB**，清理完成时为 **83.4005 GiB**；后续复核读数为 **83.3974 GiB**，净回收约 **28.93 GiB**。详细路径和逐项结果见[清理记录](../outputs/20261003_engineering_governance/runs/diagnostic-only_c_drive_cleanup_20261003/cleanup_record.json)。

清理后，Windows 环境的锁核对确认 **169 条 Conda 记录和 2 个 pip 锁包**，`pip check`、项目导入和 CPU 有限值张量 smoke 均通过；独立 CPU 环境的 **34 包锁核对**、`pip check`、项目导入和张量 smoke 均通过。两套环境检查均从各自 Python 解释器直接执行。保留的 micromamba 工具 SHA-256 与固定发布值一致，但清理后的版本/运行器探测退出码为 `-1073740791`（Windows `0xC0000409`）；原因未确定，记录为工具启动限制，不影响上述直接解释器检查结果。没有重跑完整测试或 CUDA 诊断。

### 2026-10-03 D 盘验收环境迁移与最终复核

两套验收环境已按锁文件在 D 盘独立重建并通过检查：Windows 完整锁环境为 `D:\lg-envs\win`（169条Conda记录、2个pip锁包、Shiboken归属核对、`pip check`、项目导入、模块路径隔离和CPU有限值张量检查均通过）；Windows CPU CI环境为 `D:\lg-envs\cpu`（34包哈希锁、`pip check`、项目导入、隔离路径及CPU张量检查均通过）。二者均从各自解释器直接检查，editable项目只指向 `D:\codex-original`。主环境 `D:\ANACONDA\envs\optical-generative-models-win` 未被迁移或修改。工具位于 `D:\lg-envs\tools\micromamba.exe`，SHA-256与官方固定值一致；版本和runner探测在已验证的进程上下文通过。Windows锁安装实际使用169个精确锁定归档；原始URL、SHA-256和归档验证在[归档清单](../outputs/20261003_engineering_governance/runs/diagnostic-only_d_drive_env_migration_20261003/conda_archive_manifest.jsonl)与[校验结果](../outputs/20261003_engineering_governance/runs/diagnostic-only_d_drive_env_migration_20261003/conda_archive_verification.json)中。

本地Windows Actions等价CPU工作流通过 **164个选测**，Ruff `E4/E9/F821`和`compileall`通过；第一次运行因测试子进程的临时`PATH`未含Git而失败，恢复进程原有Git可见性后同一工作流通过，失败与修正记录均保留。该结果是本地验收，不代表远端Actions已运行。

验收、缓存清理后，C盘白名单中的三个旧环境/工具根目录均已删除；D盘 `scratch` 和 micromamba包缓存目录也已清除且没有重建。C盘删除前空闲 **81.9845 GiB**，删除后立即 **90.6064 GiB**，净释放 **8.6219 GiB**；最终复核约 **90.60 GiB**。D盘由迁移开始时空闲 **198.1333 GiB**至最终约 **189.67 GiB**。删除后两套D盘环境重新通过锁校验、`pip check`、导入和CPU张量smoke；micromamba和mpmath归档哈希也重新核对。逐项目标、删除数量与空间读数见[C盘迁移清理记录](../outputs/20261003_engineering_governance/runs/diagnostic-only_d_drive_env_migration_20261003/c_cleanup_result.json)，迁移总记录见[迁移记录](../outputs/20261003_engineering_governance/runs/diagnostic-only_d_drive_env_migration_20261003/migration_record.json)，删除后路径/环境/文档最终复核见[最终验收记录](../outputs/20261003_engineering_governance/runs/diagnostic-only_d_drive_env_migration_20261003/final_verification.json)。

## 历史停止点：笔画保真盲判

笔画保真A/B对照和评分已收尾，父报告仍为`REVIEW_INCONCLUSIVE / REVIEW_EVIDENCE_INSUFFICIENT`。起点/A5400/B5400分别人眼认对100/200、103/200、106/200；真实目标49/50通过，但隐藏重复仅24/30一致，低于27/30审阅质量要求。因此不判B5400的SSIM候选有效或无效，也不把审阅不足写成模型科学失败。详见[原评分报告](../outputs/20261008_m0_generator_stroke_fidelity_comparison_v1/runs/attempt_08/report.json)。

[零更新呈现定位](../outputs/20261008_m0_generator_stroke_presentation_localization_v1/runs/attempt_01/report.json)核对三端点共1500行：保存的28×28视图可由raw逐像素重建；600张主审PNG与对应视图一致，30组隐藏重复PNG一致。平均约0.20%的raw像素超过1，平均截断强度损失占raw总强度0.28%；池化与截断顺序变化的视图平均绝对差约0.00026，局部最大差约0.274，8位量化RMSE约0.00113。没有证据表明保存/页面图像错配或截断是不可读主因。类别2/4仍显示原始笔画误差，但审阅一致性门槛未过，具体呈现机制保留未知。

变更后的评分与定位聚焦回归 **15 passed**，变更文件编译、Ruff及`git diff --check`通过。此任务没有模型更新、光学或评价器前向、梯度、采样或新增人工评分。根[AGENTS.md](../AGENTS.md)规定，人工审查留给模型采用、采样生成、物理部署和正式可读性验收等关键节点；普通工程修复、局部诊断和中间训练点不默认启动。保留A4700研究基线；不续训、不改系数、不扩Encoder、不启动采样，`training_recommendation=NONE`、`next_action=NONE`。人工可读性正式结论、独立泛化、多样性/新颖性及完整语义生成仍未完成。

冻结方向诊断的原attempt_02分数差导数遗漏最大错误类别项，故其方向点积、夹角和科学分类不采信；原报告与纠正证据均保留。修复后的v2测量见[attempt_04有效报告](../outputs/20261004_m0_generator_semantic_direction_adam_diagnostic_v1/runs/attempt_04_margin_gradient_repair/report.json)。中间的[attempt_03无效记录](../outputs/20261004_m0_generator_semantic_direction_adam_diagnostic_v1/runs/attempt_03_margin_gradient_repair/failure.json)因把人口角色摘要误当逐行角色列表而在预检失败；该工程错误已修正，没有产生科学更新或科学结论。

attempt_04通过CPU float64解析反例、独立分量梯度和中心有限差分：符号反例导数为−1、公共logit平移导数为0、精确并列按不可判定处理。三端点代表行的CUDA生产梯度均与`g_true−g_wrong`一致，最大L2误差为`4.59×10⁻⁷`；全部200行拟合人口按原路径冻结重测，父读出最大差为0、预测翻转为0，科学更新为0。最终为`VALID_COMPLETED / MIXED_OR_INSUFFICIENT`：类别1和7均没有满足16/20稳定有利条件的完整方向；保存Adam、零历史Adam与辅助梯度投影都未形成支持候选。训练建议`NONE`、`next_action=NONE`。这修复了方向测量的有效性，但没有改变父CE训练的`VALID_FAILED / NOT_SUPPORTED_WITHIN_SCOPE`，也未证明完整语义生成成功。E_D已用于方向诊断，后续确认不再是E_D盲测。

修复任务聚焦测试 **18 passed**；变更文件`compileall`、Ruff及`git diff --check`通过。全套pytest为 **2079 passed, 1 failed, 13 deselected, 25 subtests passed**；唯一失败仍是无关的`near_conjugate`诊断写证据时缺少输出父目录。全套测试在最后一个小型预检人口修复前运行；修复后受影响的聚焦测试、编译、Ruff与差异检查均已重跑通过。完整结构和入口见[Research Code Map](research/project_map.md)。

随后完成[Generator有限位移响应诊断](../outputs/20261006_m0_generator_finite_displacement_response_v1/runs/attempt_03/report.json)，任务 `M0_GENERATOR_FINITE_DISPLACEMENT_RESPONSE_V1` 为 `VALID_COMPLETED / TRAINING_DISPLACEMENT_MISMATCH_WITH_LOCAL_RESPONSE`。固定B端点、200行拟合TRAIN和2048生产光路；模型、Adam、评价器均冻结，科学更新为0。CUDA预检的8个E_D中心有限差分控制均通过，模型/评价器/Adam冻结检查通过；20个批次各完成44个预声明位移条件，共880次位移、9000行正式前向。保存Adam候选在类别1为8/20有利、10/20有害，在类别7为16/20有利；完整重置Adam候选分别为12/20和13/20有利，均未满足两类同时达到16/20的训练位移条件。作为局部响应正控，等RMS的E_D类别1和类别7梯度位移分别使对应类别20/20行的分数差增加；这说明附近存在可测响应，但不证明分类边界可被训练跨越，也暴露出该类局部位移可能损伤其他类别。结论仅适用于批次条件的有限相位干预，不是共同checkpoint准确率或可读性结果；不支持续训、重置Adam、投影或扩容，训练建议`NONE`、`next_action=NONE`。attempt_03保存了20个测量块、逐行指标和100行固定色阶图；恢复完成态未追加前向。图像报告首次收尾的维度错误及已保留记录不影响随后CPU恢复生成的有效报告。聚焦测试 **11 passed**；变更入口与测试的Windows `compileall`、Ruff及`git diff --check`通过；未重跑全套测试。

随后完成[共同Generator方向的冻结跨人口验证](../outputs/20261006_m0_generator_shared_direction_transfer_v1/runs/attempt_03/report.json)，任务 `M0_GENERATOR_SHARED_DIRECTION_TRANSFER_V1` 为 `VALID_COMPLETED / SHARED_RESPONSE_WITH_CLASS_OR_IMAGE_TRADEOFF`。固定step2500的B端点，从原200行拟合人口保存的批次梯度构造四个共同方向，并将每个相位状态用于完整600行canonical TRAIN（200行构造、100行历史比较、300行确认）；模型、Adam及评价器保持冻结，科学更新为0。READY预检、四评价器目标正控、独立基线重复（最大读数差为0）及冻结核验均通过；10个干预条件共产生6,000行读数，连同600行基线共6,600行。主判定只看300行确认人口：等RMS纯CE方向使类别1和7的E_D分数差分别在30/30行上升，同时类别8在30/30行下降；三类正确数均未改变，类别1/7仍为0/30，宏召回仍为0.30。加入BOGT或保存Adam后类别响应变化，不能从本次冻结对照单独归因。故结果是共同局部响应伴随类别代价，不满足无稳定类别伤害的支持条件；不支持续训或扩容，训练建议`NONE`、`next_action=NONE`。attempt_01/02的无效工程记录保留，只有attempt_03包含有效完整测量；入口及测试见[诊断脚本](../scripts/research/lightgen/diagnose_lightgen_2d_generator_shared_direction_transfer_v1.py)和[回归测试](../tests/research/lightgen/test_lightgen_2d_generator_shared_direction_transfer_v1.py)。本轮聚焦测试 **7 passed**，新增文件`compileall`、Ruff和`git diff --check`通过；未重跑全套测试。
