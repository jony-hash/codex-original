# 当前项目状态

更新时间：2026-10-02。执行规则以[根 AGENTS.md](../AGENTS.md)为准。

## LightGen 主线

当前问题是 BOGT 联合目标能否经统一 Encoder→复 OLS→Generator 映射产生可辨识的 CMOS 数字图像。规范梯度曾改善 DEV 和 TRAIN-neighbor S1 的独立 E_D 读数，但 S1 未达到原生成门槛。small-bank step2200 是研究证据；其原训练 attempt 仍为 `INVALID_HARNESS`，只读对账没有追认 Gate。S1 属于 TRAIN 邻域采样，不代表独立先验新颖性。

最近的类别1/7定位任务 `M0_BOGT_CLASS17_JOINT_MAPPING_ATTRIBUTION_V1` 保持 `attempt_01=INVALID_HARNESS`、参数更新为0；固定 step1200/step2200 前向读数已保存，但探针、梯度和完整报告未完成。失败的 `preflight_04` 没有启动科学测量，因此类别1/7瓶颈仍无定位结论。已停止 M0 执行器归档，不是当前可执行入口。历史见[整理前状态快照](archive/20261001_project_refactor/CURRENT_STATUS_pre_refactor.md)及[实验历史](status/experiment_history.md)。

## 工程扫尾

本轮完成 LightGen 主线及其直接依赖的小范围工程整理，覆盖现存 BOGT 训练、三评价器规范方向、S1 采样和相关加载/恢复路径。恢复不再以源码、文件摘要、dirty 状态或 Git HEAD 相等作为准入条件；实际加载的模型/配置、更新步数与 Adam、样本行与标签、数据角色、冻结范围及数值状态仍会校验。耗时保留为观测，不作为硬拒绝；硬件安全检查保留。

三评价器规范方向的共享计算位于 `src/ldm_project/v2/application/research/lightgen/canonical_semantics.py`，BOGT 训练消费位于 `src/ldm_project/v2/application/research/lightgen/lightgen_2d_chen_bogt_complex_ols_training.py`。目录测试现在检查 Git 中实际存在的配置；pytest 默认不写仓库父目录缓存，测试临时目录使用外部可写路径。停止任务和无当前消费者的研究家族没有在本轮扩展整理。

验证结果：聚焦 pytest **101 passed**；全套 pytest **1932 passed, 13 deselected, 25 subtests passed**；Windows `compileall`、`git diff --check` 通过。Ruff 对本轮变更通过；三评价器历史脚本已有的 E702 分号格式问题单独排除，未扩展成历史格式清理。

## 当前停止点

本轮只完成代码、测试入口与文档收尾，没有启动科学预检、训练、测量、恢复或 CUDA。其他研究家族和无当前消费者的历史路径未处理。当前没有科学执行授权；后续先由用户确定一个研究问题，再按现行根合同处理。

完整历史文本见[2026-10-01整理归档](archive/20261001_project_refactor/)；项目结构见[Research Code Map](research/project_map.md)。
