# Gaussian OLS 仿真基线与 LP01 对照验证

## 结果与状态

最终状态：**`COMPLETE / GAUSSIAN_S12_REFERENCE_VALIDATED_LP01_SENSITIVITY_RECORDED`**。AutoDL 最终完整测量退出码为 0，11 项预先定义的数值检查及聚合检查全部通过。Gaussian 复场投影—传输—重发射可作为当前未标定参考几何下的主仿真基线；LP01 保留为同几何模式假设对照。本结论只表示公式与数值实现自洽，不表示该 Gaussian 模式已被实测证明更接近真实光纤。

正式任务训练更新、诊断优化更新、数据集读取、任务评价器和 FID 调用均为 0；loss、正式默认模型和生产光路未改变。历史 G3 窗口收敛失败没有在本轮重测，仍为未解决状态。

**后续采用记录（2026-10-13）：** 用户已确认后续仿真采用 Gaussian S12 为主参考、同几何 LP01 为模式敏感性对照。该决定只应用于未来仿真参考，不表示已校准实物模式，也不迁移正式默认模型、旧 checkpoint 或历史功率参考。随后执行的[零更新 G3/OLS 维度复核](ols_gaussian_dimension_reaudit_v1.md)仍在 `1e-4` 复场门槛下 FAIL；因此维度响应测量被 Gate 阻断，未产生结构排序。这是该后续任务的新收敛结果，不改写本报告中 Gaussian 模式公式与单窗口数值验证通过的历史结果。

## 科学认识

Gaussian 采用本计划指定的论文 S12 形式：

\[
u_j(x,y)=\sqrt{\frac{2}{\pi w^2}}\exp\!\left[-\frac{(x-x_j)^2+(y-y_j)^2}{w^2}\right],\quad w=7.111\,\mu\mathrm m,
\]

因此连续面满足 \(\int |u_j|^2\,dA=1\)。离散复场投影显式包含像素面积元素：

\[
c_j=\sum_{m,n}u_j^*(x_m,y_n)E(x_m,y_n)\,\Delta x\Delta y,\quad \mathbf b=\mathbf T\mathbf c,\quad E_{out}=\sum_j b_jv_j.
\]

Gaussian 路径使用解析归一化的完整全局 Gaussian 模式，不逐端口裁切、离散重归一化或正交化；本轮固定 \(\mathbf T=\mathbf I_{100}\)。LP01 对照使用同一中心坐标和名义 1/e 振幅半径，由仓库 V=2.2 阶跃折射率 LP01 解构造，按原 Voronoi 单元截断并在单元内离散归一化。两者消费完全相同的复场、参考面和采样，不要求结果数值相等。

早期 20 μm pitch、7 μm MFD 的 Gaussian 基线作为独立历史案例保留，没有混入本轮 125 μm pitch 的同几何比较。

固定仿真假设如下，均不是实测器件参数：

| 参数 | 本轮取值 |
|---|---:|
| 波长、全局采样间隔 | 532 nm、2 μm |
| 相位层 | 2 层，每层 100×100、200×200 μm |
| 层间距、末层到光纤面距离 | 1.5 mm、1.5 mm |
| 编码器到光纤面的总传播距离 | 3.0 mm |
| 公共全局传播面 | 2048×2048、4.096×4.096 mm，未 resize |
| 光纤阵列 | 10×10、pitch 125 μm、包层直径假设 125 μm |
| 模场参数 | MFD 假设 14.222 μm，Gaussian 1/e 振幅半径 7.111 μm |
| OLS 系数传输 | 理想 \(T=I_{100}\)，无损耗差异、串扰或相位漂移 |
| synthetic 输入功率 | 249.3 μW，相干均匀 200 μm 方形照明 |
| 每层功率透射 | 0.889，仿真假设 |

10 个中心的 x/y 坐标均为 \([-562.5,-437.5,-312.5,-187.5,-62.5,62.5,187.5,312.5,437.5,562.5]\) μm。最外中心为 \((\pm562.5,\pm562.5)\) μm，中心跨度 1.125 mm；加上假设的 125 μm 包层后，阵列外尺寸为 1.250 mm。100 个坐标逐通道保存在机器报告中。

对三种固定 phase-only synthetic 状态，同一入射复场经过相同的生产两层 FiniteCropPhaseStack/ASM 后，再由两种 OLS 基底分别投影和重发射。传播至 OLS 前的功率约 184.676 μW。表中端面效率为 \(P_{fiber}/P_{before\ OLS}\)，系统效率为 \(P_{fiber}/249.3\,\mu W\)：

| 固定相位状态 | 模型 | 捕获功率 | 端面效率 | 系统效率 | 单场 \(N_{eff}\) | 边缘/中心平均通道功率 |
|---|---|---:|---:|---:|---:|---:|
| 零相位 | Gaussian | 9.35072 μW | 5.0633% | 3.7508% | 4.0000 | 1.064×10⁻⁹ |
| 零相位 | LP01 | 11.95064 μW | 6.4711% | 4.7937% | 4.0000 | 1.088×10⁻⁹ |
| 末层 x 向 1° | Gaussian | 2.43626 μW | 1.3192% | 0.9772% | 3.4168 | 1.285×10⁻⁹ |
| 末层 x 向 1° | LP01 | 2.19351 μW | 1.1878% | 0.8799% | 3.3927 | 1.266×10⁻⁹ |
| 末层 x/y 各 1° | Gaussian | 0.63466 μW | 0.3437% | 0.2546% | 2.9182 | 3.211×10⁻⁹ |
| 末层 x/y 各 1° | LP01 | 0.52844 μW | 0.2861% | 0.2120% | 2.8963 | 3.209×10⁻⁹ |

零相位状态下 LP01 的 6.4711% 端面效率与历史 synthetic 诊断约 6.47% 一致；在同场上 Gaussian 为 5.0633%。该场的功率参与率仍约为 4，而且边缘/中心功率比约为 \(10^{-9}\)。这说明该**未优化 synthetic 场**高度集中于中心通道；它既不证明 Gaussian 模型错误，也不证明经过优化的 phase-only 系统无法照亮外侧 fiber。本轮没有开展 fiber 可达性或阵列优化，不能据此作 100 维潜空间决策。每个状态的完整 100 通道功率向量、分位数和复系数都保存在 JSON 与 `.pt` 读出中。

同一复 latent 的 LP01 相对 Gaussian 误差按共同全局相位计算：\(\phi=\arg(\sum_j\overline{c_{G,j}}c_{L,j})\)，随后仅将整个 LP01 向量乘 \(e^{-i\phi}\)。没有逐通道相位拟合。

| 状态 | raw 复 latent 相对误差 | 共同相位对齐后 | 共同相位 \(\phi\) | 重发射强度相对误差 | 通道功率分布相对误差 |
|---|---:|---:|---:|---:|---:|
| 零相位 | 13.2118% | 13.0506% | −0.01935 rad | 24.9565% | 27.8044% |
| 末层 x 向 1° | 5.4235% | 5.3225% | −0.01069 rad | 12.5279% | 9.6786% |
| 末层 x/y 各 1° | 8.8546% | 8.7857% | −0.01154 rad | 18.8921% | 16.4317% |

每根 fiber 的复 latent 误差另保存 raw 与共同相位对齐两组数组，均以 \(\sqrt{P_{input}}\) 归一化。逐通道对齐误差最大值分别为 0.012638、0.002856、0.003019。独立 NumPy 复算确认共同相位和两组逐通道数组与报告一致；这三组模式差异应解释为 Gaussian/LP01 基底假设敏感性，不是数值回归失败。

中心通道定义为偶数行列阵列中最近中心的四根；edge/center 比值是外圈 36 根的平均功率除以这四根的平均功率。CPU 解析参考的复相位直接比较，未做 global-phase alignment；全局相位对齐只用于 Gaussian 与 LP01 的模型间比较。

## 目标缺口

所有光纤几何、MFD、NA、包层尺寸、光纤参考面和 0.889 透射值均为未测量仿真假设。理想 \(T=I\) 未包含真实通道损耗、串扰、相位漂移、偏振及器件差异。因而目前可以验证 Gaussian 公式及实现内部自洽，却不能声称其比 LP01 更符合某个实际 fiber array，也不能给出硬件可实现性结论。

本轮没有 task-level 生成任务、语义评价、相机标定、fiber 数量/布局优化或非线性模型。单场 \(N_{eff}\) 是功率参与率，不是独立复响应秩；也未接入未经物理解释的紧凑 Decoder 映射。历史 G3 窗口变化下的复场收敛 FAIL 未复测，仍须单独处理。

## 工程完成与验证

最终运行环境为 AutoDL `autodl-monthly`：Python 3.12.3、PyTorch 2.5.1+cu124、CUDA 12.4、RTX 4080。代表性生产前向预检与最终完整运行均退出码 0；最终报告 12/12 个检查为 true，运行观测约 10.75 s、峰值显存分配约 702 MB。预检和完整测量的正式训练、phase optimizer 更新、dataset/evaluator/FID 调用均为 0。

| 检查 | 结果 |
|---|---:|
| Gaussian/LP01 离散模式范数最大误差 | 2.04×10⁻¹⁴ / 3.88×10⁻⁹（门槛 10⁻⁴） |
| 完整投影—重发射功率增益上界 | Gaussian 1.0000000000000413；LP01 0.999999992236（相对被动容差 10⁻⁵） |
| 3 个 phase case 的 GPU/独立 CPU 复系数误差最大值 | Gaussian 2.92×10⁻⁸；LP01 8.88×10⁻⁹（门槛 10⁻⁴） |
| 中心最近端口、边中端口和四角单模匹配 | 24 项通过，最大误差 1.29×10⁻⁷ |
| 双端口相干相位 0、π/2、π | 6 项通过，最大相位误差 8.75×10⁻⁸ rad |
| 有限 200 μm 方形场的 x/y 轴 0、±1、±3、±5° 倾角 | 14 项通过；Gaussian/LP01 最大复系数误差 1.92×10⁻⁸ / 8.51×10⁻⁹ |
| 负控：丢弃复相位、遗漏像素面积 | 误差 0.0856、1.302×10¹⁰，均被 10⁻⁴ 门槛拒绝 |
| 重叠 Gaussian 模式负控 | 1 μm 间距时增益 3.96074，被被动性检查拒绝 |
| 独立 mpmath LP01 本征解 | 与生产解最大绝对差 4.44×10⁻¹⁶ |
| Autograd 与中心有限差分 | 固定步长 0.03 rad；相对误差 1.4824×10⁻⁴ / 6.6350×10⁻⁵，均低于原 2×10⁻³ 门槛 |

梯度检查初次使用 0.001 rad 时第一层误差为 8.67×10⁻³。随后在同一 GPU/复数精度下完成 0.001、0.003、0.01、0.03 rad 扫描及 0.02–0.05 rad 邻域复核；0.02–0.05 rad 的全部结果都低于 2×10⁻³。最终只把有限差分步长固定为 0.03 rad，没有放宽容差。较小步长在 complex64 前向下受到差分消减影响。

实现过程中的工程问题均以新 overlay/new attempt 保留，没有覆盖旧结果：修正了 fresh process 下无效的 CUDA 计数器 reset、Gaussian/LP01 重发射张量维数处理、NumPy bool JSON 转换，以及 per-fiber 误差的共同相位定义。最终 per-fiber raw/aligned 结果经独立 NumPy 重算核对。早期 attempt 的失败都属于测量/报告 harness 修复，不作为科学负结果；最终结论只取修复后的 attempt_05。

最终 AutoDL 原始证据已按精确文件清单回传，收据为 `COMPLETE`：14 个文件共 303,615,511 bytes，包含完整复场读出（约 289 MiB）、相位屏、复系数及通道功率、Gram 矩阵、CPU/梯度核验、机器报告、执行配置/runner 和资源遥测。attempt_03 历史梯度报告及两组梯度步长诊断另有 3 份完整回传收据。交付审计加载了复场张量（complex64，[1,1,2048,2048]）和 100×100 Gram 矩阵，并验证所有回传文件大小。

资源遥测采集失败为 0；短诊断只留下 1 个采样点，作为运行观察，不用于推断持续负载。

主要证据：

- [最终机器报告](../outputs/20261013_ols_gaussian_lp01_model_alignment_v1/runs/autodl_backup_attempt_05/report.json)、[云端报告副本](../outputs/20261013_ols_gaussian_lp01_model_alignment_v1/runs/autodl_backup_attempt_05/report.md)
- [原始复场读出](../outputs/20261013_ols_gaussian_lp01_model_alignment_v1/runs/autodl_backup_attempt_05/complex_field_readouts.pt)、[系数和逐 fiber 功率](../outputs/20261013_ols_gaussian_lp01_model_alignment_v1/runs/autodl_backup_attempt_05/coefficients_and_power.pt)、[Gram 矩阵](../outputs/20261013_ols_gaussian_lp01_model_alignment_v1/runs/autodl_backup_attempt_05/gram_matrices.pt)
- [最终回传收据](../outputs/20261013_ols_gaussian_lp01_model_alignment_v1/runs/autodl_backup_attempt_05/transfer_receipt.json)、[独立逐 fiber 指标交付审计](../outputs/20261013_ols_gaussian_lp01_model_alignment_v1/runs/delivery_audit_01/delivery_audit.json)
- [初始梯度步长扫描](../outputs/20261013_ols_gaussian_lp01_model_alignment_v1/runs/autodl_backup_gradient_resolution_01/gradient_step_sweep.json)、[稳定邻域复核](../outputs/20261013_ols_gaussian_lp01_model_alignment_v1/runs/autodl_backup_gradient_resolution_02/gradient_step_neighborhood.json)

本轮全项目未跑 pytest；对任务 runner/config 执行了 Ruff、Python AST/YAML 解析、云端 py_compile/CLI、代表性 GPU 前向、全部数值 Gate 和回传后独立复算。

## 决策影响与下一步

建议后续仿真以 Gaussian S12 作为统一主参考，保留同几何 LP01 作为模式假设敏感性对照。依据是 Gaussian 的归一化、坐标、复相位、功率、被动性和独立参考全部闭合，而不是因为它与 LP01 数值相等或已经证明与实物更吻合。正式运行默认模型保持不变。

下一阶段可单独规划 OLS 维度研究：同时看名义通道数、单场功率参与率、独立 synthetic 探针下的复响应秩/奇异值/可控性，并分别保留 Gaussian 与 LP01 的敏感性比较。只有结构选择之后再讨论非线性分布。以上是建议；本轮不启动维度扫描、非线性、正式训练、数据读取或 evaluator。历史 G3 收敛问题仍需单独获得测量结论。
