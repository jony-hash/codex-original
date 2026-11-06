# 全量 CPU 验证记录（2026-11-07）

## 状态

本地全量已完成；云端待 Codex 执行。只记录，不修复。

## 命令与环境

- `python scripts/run_windows_cpu_checks.py --full`
- Windows 解释器：`D:\ANACONDA\envs\optical-generative-models-win\python.exe`
- 采集 3562 个测试、13 个 deselected。

## 本地结果

```
16 failed, 3525 passed, 1 skipped, 13 deselected, 1 warning, 7 errors,
25 subtests passed in 805.02s (0:13:25)
```

## 失败分布

| 区域 | 失败 | 错误 |
| --- | --- | --- |
| `tests/research/lightgen` | 8 | 7（`LightGenPostTrainingEvaluationContractTests` setup） |
| `tests/research/scaled_chen` | 6 | 0 |
| `tests/research/capacity` | 2 | 0 |

失败清单：

```
tests/research/capacity/test_controller_semantic_geometry_data_firewall.py (2)
tests/research/lightgen/test_class_balanced_joint_canary.py (1)
tests/research/lightgen/test_define_lightgen_2d_semantic_complex_ols_training_contract_v1.py (1)
tests/research/lightgen/test_define_lightgen_2d_semantic_generation_input_ols_alignment_contract_v1.py (1)
tests/research/lightgen/test_lightgen_2d_establish_sic_independent_confirmation_evaluator_v1.py (1)
tests/research/lightgen/test_local_compute_bounded_joint_canary_contract.py (1)
tests/research/lightgen/test_local_compute_bounded_joint_canary_executor.py (3)
tests/research/scaled_chen/test_scaled_chen_controller_lineage_migration.py (2)
tests/research/scaled_chen/test_scaled_chen_controller_state_authority.py (3)
tests/research/scaled_chen/test_scaled_chen_output_semantic_evaluator_requalification.py (1)
```

## 根因

- 缺失 dated 工件/检查点：`outputs/20260812_v2/.../conditional_optical_generator.pt`、`outputs/20261106_autodl_storage_cleanup_v11/.../cloud_archive/...` 等，报 `FileNotFoundError`。
- 其余为 contract 断言与历史 authority 不匹配。
- **与本次 legacy 归并无关**：失败日志中引用被移动目录（`_legacy_2026q3`、`reports/`、`ppt_*` 等）的次数为 0；引用的是未移动的 dated `outputs/2026*` 路径。

## Harness 缺陷（新发现，未修）

测试跑到 100% 后，`run_windows_cpu_checks.py` 在清理 `tempfile.TemporaryDirectory` 时报
`OSError: [WinError 145]`（长路径目录非空），脚本以 traceback 结束。后果：

- 退出码不可靠（本次由外层 shell 吞掉）；
- 运行结果只存在于日志，脚本没有打印自己的 verdict。

修复方向（下一批，不属本次）：改用短路径 basetemp 或对长路径目录做 `ignore_errors` 清理。

## 云端结果（AutoDL）

待 Codex 执行，步骤见 [`docs/operations/full_suite_autodl.md`](../../operations/full_suite_autodl.md)。回传日志后补填此节。

## 附录：`docs/archive/` 既有断链

794 条，分布在 4 个文件，全部指向已不存在的 `outputs/` 或旧路径；归档条目不可改写，本轮只统计不修复。