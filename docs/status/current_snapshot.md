# 当前快照（2026-11-06）

> 这一页只回答四件事：主线是什么、停在哪、允许做什么、还缺什么。日期细节见 [CURRENT_STATUS](../CURRENT_STATUS.md)，停止点原文见 [current_handoff](current_handoff.md)，路线清单见 [route_status](../research/route_status.md)。

## 1. 活跃路线

- **正式 v1**：`train.py` 驱动的 LDM teacher 蒸馏线，保留运行，不是当前默认主线。
- **正式 v2**：`python -m ldm_project.v2`，teacher-free conditional runtime；`application/current/` 是唯一 formal 实现。
- **LightGen OLS-native**：当前活跃研究线，最新工作围绕 HW207 物理映射与 C3 系列数值收尾。
- 其余研究家族（scaled_chen、optical_manifold/fiber、capacity）的状态与禁止事项见 [route_status](../research/route_status.md)。

来源：[v2 runtime](../architecture/v2_runtime.md)、[research code map](../research/project_map.md)、[repository taxonomy](../repository_taxonomy.md)。

## 2. 当前停止点（2026-11-06）

- HW207 **C3.8**：`C3.8_PARTIAL_PASS`。单一共享相位屏在开发验证集上 Accuracy 91.05%，但类别峰值路由未达 90%，种子间能量保持不稳定；正式训练未启用 `spread`，五项联合目标只经过 P2 预检和 T4 pilot。
- HW207 **C3.7**：`C3.7_PARTIAL_PASS`。现有候选未同时满足分类、能量和核心聚焦目标；E4/E5 的阶段绑定问题已定位。
- **G1**：`LIGHTGEN_G1_FIXED_PHASE_TEACHER_DISTILLATION_V1` 保持 `diagnostic-only`，教师资格为 `TRAINING_REFERENCE_INSUFFICIENT`；未启动学生训练，科学更新为 0。
- 三个阻断状态维持：`HW207_CURRENT_MAPPING=BLOCKED`、`INDEPENDENT_EVAL=BLOCKED`、`FABRICATION_READY=BLOCKED`。

来源：[C3.8 交接](current_handoff.md)、[C3.8 报告](../research/lightgen/hw207_mnist4_c38_report.md)、[CURRENT_STATUS 最新条目](../CURRENT_STATUS.md)。

## 3. 允许动作

- 允许：读取、重放和审计已保存证据；在明确授权的新任务内执行训练、测量或恢复。
- 默认不允许：追加训练、开启独立评估、加工放行、实物打印、硬件实验；这些需要显式授权，交接中的"建议"不构成授权。
- 新运行路径与证据绑定遵循 [output_artifact_notes](../operations/output_artifact_notes.md)；正式 CUDA 验收的缺失输入视为 blocker。

## 4. 未验证缺口

- **实物映射**：DMD 身份/方向/收光、输入与输出 Relay 实测映射、CMOS 实机参数与像素到 ROI 标定、重复采集稳定性。
- **材料与加工**：打印层 532 nm 复透射、材料 Δn、打印机横向响应、最小特征与加工鲁棒性；现有加工结论均为假设算子下的数值诊断。
- **评估人口**：官方 test 关闭；开发验证集已用于选模，尚未建立项目级独立评估人口。
- **G1 教师**：固定零相位下未取得合格教师；这不证明其他相位或整个分布不可达。

## 维护规则

本页只做摘要，不新增判定。`CURRENT_STATUS.md` 与 `current_handoff.md` 的日期条目仍是证据正文；两者冲突时以正文为准并修正本页。更新本页时同步更新时间与来源链接。