# Gaussian/OLS Decoder 可微链与三步更新验收 v11

运行日期：2026-10-18。数值工作运行目录为 AutoDL `outputs/20261017_ols_decoder_autograd_update_preflight_v11/runs/attempt_01`；精确回传副本位于[`autodl_backup_v11_return`](../../../outputs/20261017_ols_decoder_autograd_update_preflight_v11/runs/autodl_backup_v11_return)。机器结果见[JSON artifact](../../../artifacts/ols_decoder_autograd_update_preflight_v11.json)，主包111/111文件匹配；首次失败任务的控制日志另存于[`autodl_backup_v11_failure_logs`](../../../outputs/20261017_ols_decoder_autograd_update_preflight_v11/runs/autodl_backup_v11_failure_logs)，5/5文件匹配。

## 结果与状态

最终状态为 **`ENGINEERING_ACCEPTED`**，AutoDL任务 `ols-dec-v11-three-step-repair` RunId `690cf3570afd4d70a2233dd748a64b98`，退出码0。2×2-C和3×3-B都通过完整Gaussian S12→4f→两层Decoder→后端成像→W/pixel→代理相机损失的值、梯度、三步更新、窗口和独立恢复检查。实际共调用10次工程Adam：每候选连续3步、从step 1独立恢复后再走2步。正式科学训练更新、数据集读取、任务评价器/FID、材料前向和loss修改均为0。

预检最初因2×2-C第一层归一化负梯度方向的差分误差而阻止更新。把中心差分步长从`1e-5/5e-6 rad`细化到`1e-6/5e-7 rad`后，该项误差降至`7.68e-8/1.20e-8`，低于未变更的原容差；两候选其余梯度方向也均通过。此调整只改变有限差分测量步长，没有放宽门槛。

## 科学认识

**科学判断未改变。** 本轮证明的是限定仿真参数和冻结输入下的工程可微链与恢复能力；不证明重建质量、语义图像质量、OLS可用维数、真实相机性能或实验光路有效性。Gaussian S12仍是仿真主参考，LP01仅为同相位敏感性对照。

3步工程目标值随更新步下降：2×2-C为`0.0475353 → 0.0471790 → 0.0468244`，3×3-B为`0.00124750 → 0.00124092 → 0.00123437`。它们是代理相机任务输出上的固定工程MSE，不是训练收益或图像质量指标。

1 ms代理曝光的终点仍显示显著动态范围差异：2×2-C在Gaussian下均匀输入/最坏二值图的饱和像素比例为`94.37%/94.49%`，LP01约`94.78%/94.86%`；3×3-B在这两种模式和三类终点输入上均未饱和，但其传感器端平均像素功率约为`8.4e-13 W/pixel`量级，明显低于四通道约`5.0e-10 W/pixel`量级。曝光和相机参数是仿真假设，未据此选曝光、gain或OLS维数。

## 目标缺口

仍缺真实光纤传输与模式参数、真实Decoder/4f及相机标定，以及任务目标下的重建和语义评价。九通道在代理相机中未饱和，不足以证明信噪比、可观测性或语义信息充足；四通道饱和也不表示其光学结构失败。旧Generator恢复验收仍为`INVALID_HARNESS / INCOMPLETE`、历史39/45次工程更新，与本轮独立。

## 工程完成与验证

- 两候选初始W/pixel与v9保存结果的相对L2误差约`4.99e-7–5.11e-7`；相机任务输出误差约`4.37e-7–5.14e-7`。GL4与GL8像素积分差最大约`1.02e-6`。
- 每层固定随机方向与归一化负梯度方向均通过complex128中心差分；边界单侧导数、共同相位活塞、双层有限非零梯度和全部负控通过。负控覆盖detach消费者、漏像素面积、把功率透射误作振幅透射、光纤端口乱序和逐样本最大值归一化。
- step 0–3、两输入图样、4096/6144/8192网格的complex64共同ROI两两比较全部通过`1e-4`；连续轨迹最大误差2×2-C为`4.51e-6`、3×3-B为`9.50e-6`。终点既有最坏二值探针也通过，complex128和GL8读数作为补充检查。
- 两候选各自的连续与恢复终点相位、Adam张量、RNG、历史记录差异均为0；相机输出相对L2误差为0。连续与恢复路径的窗口检查均通过。
- 首次完整入口在2×2-C连续三步完成后，恢复子进程因按脚本文件路径启动导致`ModuleNotFoundError: No module named 'scripts'`而退出；该次没有执行恢复更新。screen退出状态、主运行日志和遥测见补充回传目录；首次子进程stderr的原文件随后被成功重试覆盖，错误文本来自失败后实际读取记录，修复前脚本另保存在主回传包。已修为从项目根目录以`python -m scripts.research.lightgen.execute_ols_decoder_autograd_update_preflight_v11`启动，并加入对已完成连续轨迹及检查点身份/配置/窗口结果的显式校验，避免重跑已完成的3步。修复后预检模块入口成功，恢复与总计10次调用完整通过。
- 本地`py_compile`、定向Ruff和聚焦pytest通过（3项）；云端编译与模块`--help`通过。精确回传收据状态`COMPLETE`，111个远端文件逐项大小核对无差异。资源遥测采集2条、采集失败0；数值只作运行观察。

## 决策影响与下一步

四通道和九通道均取得本计划范围内的工程资格，可据此**另行制定**使用同一可微入口的短程Decoder重建对照计划，并预先处理1 ms代理相机下四通道饱和、九通道低功率的问题。本轮没有选择曝光、OLS维数或实验结构，也没有授权启动下一轮重建训练；后续计划需独立确定训练人口、固定读出规则、更新预算和图像质量判据。材料研究、正式生成训练和旧Generator断点恢复仍各自独立。

## 本地状态

新增v11配置、研究CLI、聚焦测试、报告、artifact及本轮回传证据；并按计划定点更新`docs/CURRENT_STATUS.md`、`docs/status/current_handoff.md`和Gaussian/OLS路线图。工作区原有其他未提交修改保持原状。当前分支：`research/lightgen-ols-native-semantic-generation-v1`。本轮没有提交或切换分支。
