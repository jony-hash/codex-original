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

## Harness 缺陷（已修）

测试跑到 100% 后，`run_windows_cpu_checks.py` 在清理 `tempfile.TemporaryDirectory` 时报
`OSError: [WinError 145]`（长路径目录非空），脚本以 traceback 结束，退出码不可信。

已修：basetemp 前缀缩短为 `ldm-pytest-`，并加 `ignore_cleanup_errors=True`，清理失败不再覆盖真实退出码。修复后 focused lane 仍为 176 passed / exit 0，且 `C:\TEMP` 无残留 `ldm-pytest-*`。

## 云端结果（AutoDL）

**未完成：运行在 5% 附近停滞 20 分钟以上，已终止；不构成代码判定。**

执行记录：

- 云端 GitHub 直连克隆过慢（3 分钟约 16 MB），改用增量 bundle：本地 `git bundle create` 取 `b9d865e..262a4a1`（33.4 MB），scp 31 秒传到 `/root/autodl-tmp/`，在云端仓库 fetch 后以 `git worktree add --detach` 检出到 `/root/autodl-tmp/fs-ws`（HEAD `262a4a1`，含 `--full`）。
- 云端环境：Python 3.12.3、pytest 9.0.3、`ldm-project` editable 指向旧树；用 `PYTHONPATH=/root/autodl-tmp/fs-ws/src` 覆盖。ruff 与 pymupdf 原缺失，已安装（0.17.4 / 当前版本）。
- 第一次运行在收集阶段失败：`ModuleNotFoundError: No module named 'pymupdf'`（2 个 TE 测试）。
- 安装 pymupdf 后第二次运行到约 5% 后停滞：无子进程、进程停在 eventfd 等待，宿主 load average 约 28；`py-spy` 因容器缺少 `SYS_PTRACE` 无法取栈。
- 终止进程；`/root/autodl-tmp/fs-ws` 工作树保留供重跑，磁盘剩余约 1.3 GB。

建议重跑方式：在宿主负载低时执行，加 `--timeout`（若安装 pytest-timeout）或 `timeout 1800` 外层看门狗；仍用 `PYTHONPATH=<worktree>/src`，并把日志回传到本文件。

## 附录：`docs/archive/` 既有断链

794 条，分布在 4 个文件，全部指向已不存在的 `outputs/` 或旧路径；归档条目不可改写，本轮只统计不修复。