# 当前项目状态

更新时间：2026-10-02。研究执行政策以[根 AGENTS.md](../AGENTS.md)为准。

## 主线

LightGen 当前研究问题是：BOGT 联合目标和规范语义方向能否通过统一光学映射产生可辨识的 CMOS 数字图像。规范梯度训练带来过 DEV 与 TRAIN-neighbor S1 的 E_D 改善，但 S1 仍未达到原生成门槛。此前 small-bank step2200 保存了改善证据，原训练 attempt 仍为 `INVALID_HARNESS`，只读对账不追认 Gate，也不证明完整生成已解决。

## 最近任务

`M0_BOGT_CLASS17_JOINT_MAPPING_ATTRIBUTION_V1` 的 `attempt_01` 保持 `INVALID_HARNESS`，更新数为 0。固定的 step1200/step2200 前向读数已保存，但探针、梯度和完整报告没有完成；类别1、7的语义瓶颈仍未定位。失败的 `preflight_04` 未启动科学测量。已结束或停止的 M0 执行器和对应测试移至[日期归档](../scripts/research/archive/project_refactor_20261001/lightgen/m0/)，不作为当前执行入口。详细历史与证据入口见[整理前状态快照](archive/20261001_project_refactor/CURRENT_STATUS_pre_refactor.md)。

## 工程整理

本次整理将研究脚本、运行模块、配置和测试按 `lightgen`、`scaled_chen`、`optical_manifold`、`capacity` 分组；跨组运行控制放在 `shared`。v1/v2 正式入口保持原路径。E_D/E_C 网络定义集中到 LightGen 共用模块，旧训练命令仍兼容导出同一类。传播缓存的 inference-mode 污染已修复并单独提交。当前 BOGT 数据和运行恢复比较不再因来源文件摘要变化而拒绝；摘要仍记录为 provenance。研究执行规则已收拢到根 AGENTS：不设墙钟准入上限；模型、优化器、样本行、数据角色、冻结范围、数值状态和硬件安全仍需验证。

## 当前停止点

本轮只做代码、目录和文档整理及软件验证，不运行科学训练、研究预检或 CUDA。完成本轮工程交付后，没有自动后继实验；下一项科学工作需由用户单独指定。

完整历史文本保存在[`20261001_project_refactor`归档](archive/20261001_project_refactor/)，状态导航见[experiment history](status/experiment_history.md)。
