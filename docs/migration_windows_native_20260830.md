# Windows 原生迁移记录

日期：2026-08-30

## 目标与基线

本次迁移将 `/home/dingzhen/projects/codex-original` 迁移为 Windows 主工作区 `D:\codex-original`，基线为分支 `research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`、提交 `019b76ef911e0bd16b8d5d4169807323427684af`。源项目工作树在冻结时干净，远程为 `https://github.com/jony-hash/codex-original.git`。

## 备份

备份目录：`D:\LDM-Archive\codex-original\backup`

- `codex-original-all-refs.bundle`：Git bundle，验证通过，包含全部本地 refs。
- `wsl-codex-history.tar`：WSL `~/.codex` 旧会话归档，仅作历史保存，不覆盖 Windows `%USERPROFILE%\.codex`。
- `wsl-user-config.tar`：`.ssh`、`.config`、`.claude`、`.gemini`、`ai-workspace`、shell 配置等用户配置归档。
- `wsl-conda-envs.txt`、`wsl-conda-optical-generative-models.yml`：环境清单和可重建导出。
- `migration-baseline.txt`、`backup-manifest.json`：冻结时的系统、Git、GPU、容量和备份校验记录。

归档未包含可重建的 Miniforge、缓存、VS Code Server 和包缓存。WSL 配置归档中未发现 `.local/bin` 和 `.zshrc`，因此未将不存在的路径伪装为已备份内容。

## Windows 工作区与环境

- 仓库从 Git bundle 建立，恢复原 `origin`，并设置 `core.autocrlf=false`、`core.eol=lf`、`core.longpaths=true`。
- 环境：`D:\ANACONDA\envs\optical-generative-models-win`。
- 依赖版本：Python 3.11.16、PyTorch 2.5.1、torchvision 0.20.1、CUDA 12.1、NumPy 2.2.6、Matplotlib 3.10.8、Pydantic 2.11.7、Pytest 9.0.3、Ruff 0.15.12、PyYAML 6.0.3。
- editable install 指向 `D:\codex-original`。
- Miniforge 安装器因静默安装器行为在 `C:\Users\24567\miniforge3` 建立了 bootstrap；它不属于项目工作树，也未删除现有 Conda 环境。

直接调用环境内 Python 时需确保该环境的 `Library\bin`、`Scripts` 和 `bin` 位于 PATH 中；通过 PowerShell 正常激活 Conda 环境即可满足这一条件。

## outputs 归档

归档目录：`D:\LDM-Archive\codex-original\outputs`

- 一级目录：163
- 文件数：33,117
- 源数据表面大小：240,556,414,616 字节
- tar 总大小：240,625,274,880 字节
- manifest：`outputs-manifest.jsonl`
- 索引：`outputs-index.json`

每个一级目录单独生成未压缩 tar，完成后执行目录读取和 SHA-256 校验，再处理下一个目录。历史输出未展开到 `D:\codex-original\outputs`，Git 已跟踪的正式产物保持普通文件形态。已对含尾随句点目录的归档抽样解包，路径和文件均可恢复。

## 已完成验证

- `python -m compileall src tests`：通过。
- `ruff check src tests scripts`：通过。
- CUDA 学生模型 forward 测试：正确设置 Conda DLL PATH 后 9/9 通过，设备为 RTX 3060 Laptop GPU。
- 学生产物和训练诊断测试：22/22 通过。
- 研究生命周期归档子集：8/8 通过。
- 三组历史归档契约测试：10/10 通过。
- `python -m ldm_project.v2 --help`：通过。
- CUDA smoke：训练 2 个 one-shot step，并完成 epoch 验证；输出包含 run metadata、training history、光学场张量、诊断 JSON 和 PNG。

完整默认 pytest 结果：`1010 passed, 9 failed, 3 deselected, 25 subtests passed`，用时约 3 分 54 秒。9 个失败全部是测试直接读取的历史 WSL 输出未展开：1 个 v1 teacher checkpoint，以及 8 个 20260827/20260828 v2 的历史 training/audit JSON。对应内容已在 `outputs-manifest.jsonl` 和独立 tar 中完成 SHA-256 校验；按本次迁移方案不将这些历史文件展开到 Windows 工作树，因此将其记录为归档布局限制。当前代码、依赖、CUDA 和 smoke 运行验证均通过。

## WSL 回滚窗口

Windows 验证完成后，WSL 仅保留为回滚源，窗口为 2026-08-30 至 2026-09-06。已尝试停止 Ubuntu，但当前打开的 VS Code WSL 连接会自动重新启动它；因此完全释放 WSL 内存前，需先关闭该 WSL 窗口，再执行一次 `wsl --terminate Ubuntu` 或 `wsl --shutdown`。在此期间不得在 WSL 项目上开发、训练或生成新输出。若 Windows 出现阻断问题，停止 Windows 工作区并恢复 WSL；保留失败日志和两个副本。窗口结束后，只有在用户再次明确批准后才可执行 Ubuntu 注销流程；不得手动删除 `ext4.vhdx`。
