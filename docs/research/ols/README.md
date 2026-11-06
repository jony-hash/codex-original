# OLS 历史诊断报告索引

本目录收纳 OLS 窗口、维度、相位与物理功率系列的历史诊断报告。这些报告是**历史证据与溯源**，不是当前执行入口；当前结论与允许动作见 [`CURRENT_STATUS.md`](../../CURRENT_STATUS.md) 与 [`current_handoff.md`](../../status/current_handoff.md)。

## 窗口 / 维度 / 相位闭合

| 报告 | 主题 |
| --- | --- |
| [v3](ols_g3_second_hop_error_isolation_v3.md) | G3 第二跳误差隔离 |
| [v4](ols_g3_reference_closure_v4.md) | G3 参考闭合（原 2048 窗口，FAIL 保留） |
| [v5](ols_window_dimension_power_v5.md) | 窗口维度与功率 |
| [v6](ols_phase_dimension_power_v6.md) | 相位维度与功率 |
| [v7](ols_frozen_phase_controllability_v7.md) | 冻结相位可控性 |
| [v8](ols_four_nine_window_closure_v8.md) | 四通道—九通道窗口闭合 |
| [v9](ols_decoder_camera_interface_v9.md) | Decoder—相机接口 |
| [v10](ols_camera_training_interface_closure_v10.md) | 相机训练接口闭合 |
| [v11](ols_decoder_autograd_update_preflight_v11.md) | Decoder 自动微分更新预检 |
| [v12](ols_decoder_mnist_reconstruction_v12.md) | Decoder MNIST 重建 |

## Gaussian 参考

| 报告 | 主题 |
| --- | --- |
| [dimension re-audit v1](ols_gaussian_dimension_reaudit_v1.md) | 零更新 G3/OLS 维度复核 |
| [LP01 model alignment v1](ols_gaussian_lp01_model_alignment_v1.md) | Gaussian S12 / LP01 模式对齐 |
| [zero-update closeout v2](ols_gaussian_zero_update_closeout_v2.md) | 零更新收尾 |

## 物理功率

| 报告 | 主题 |
| --- | --- |
| [power audit](physical_power_audit.md) | 14 项物理功率审计 |
| [power integration v1](physical_power_integration_v1.md) | 物理功率集成 |
| [reachability and convergence v2](physical_reachability_and_convergence_v2.md) | 物理可达性与收敛 |

## 紧凑系统设计

| 报告 | 主题 |
| --- | --- |
| [200 µm 维度优化](compact_200um_ols_dimension_optimization_report.md) | 紧凑 200 µm OLS 维度优化 |
| [架构缩放](compact_ols_architecture_scaling_report.md) | 紧凑 OLS 架构缩放扫描 |
| [Pareto 设计](compact_ols_pareto_design_report.md) | 紧凑 OLS Pareto 设计 |

## 约定

- 本目录内的相对链接以本目录为基准；重新组织时必须同步更新入链。
- 新增 OLS 报告放入本目录，并在本索引登记；不要再写回 `docs/` 根目录。