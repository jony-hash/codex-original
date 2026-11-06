# 全量 CPU 验证：AutoDL 交接

本页说明如何在云端跑一次 `--full`，作为 Windows 本地全量结果的补充证据。它不是科学验收，也不改变任何路线状态。

## 前提

- 本地提交已推送或已同步到云端副本；云端路径 `autodl-monthly:/root/autodl-tmp/codex-original`。
- 云端 Python：`/root/miniconda3/bin/python`。
- 由 Codex 在沙箱外执行 SSH；DSH 不建立远程连接，只维护本页。

## 步骤

```bash
ssh autodl-monthly
cd /root/autodl-tmp/codex-original
git fetch origin
git checkout <本地已推送的 commit>
/root/miniconda3/bin/python -c "import ldm_project; print(ldm_project.__file__)"
/root/miniconda3/bin/python scripts/run_windows_cpu_checks.py --full 2>&1 | tee /root/autodl-tmp/full_suite_<commit>.log
```

- 不覆盖云端工作树，不提交、不推送；只读运行。
- 若云端缺少依赖，先记录缺失项，不在本轮补装整套环境。

## 回传

```powershell
scp autodl-monthly:/root/autodl-tmp/full_suite_<commit>.log C:\TEMP\
```

把失败清单汇总进 [`docs/governance/20261107/full_suite.md`](../governance/20261107/full_suite.md) 的"云端结果"小节，注明 commit、时间、环境与失败类别。

## 预期差异

Linux 与 Windows 的路径、字体、内存和部分 torch 行为可能不同；云端失败不等于本地失败，反之亦然。两条结果都只作证据，不自动触发修复。

## 实测（2026-11-07）

- GitHub 直连克隆过慢（3 分钟约 16 MB）；改用增量 bundle：本地 `git bundle create inc.bundle <cloud-head>..<target>`，scp 后在云端仓库 `git fetch`，再 `git worktree add --detach` 检出，不动云端脏工作树。
- 云端缺 `ruff` 与 `pymupdf`，需先安装；`ldm-project` 的 editable 指向旧树，用 `PYTHONPATH=<worktree>/src` 覆盖。
- 本次在约 5% 处停滞 20 分钟以上（宿主 load average 约 28，无子进程），已终止。重跑请加外层 `timeout` 看门狗，并在负载低时执行。
- 详细记录见 [full_suite.md](../governance/20261107/full_suite.md)。