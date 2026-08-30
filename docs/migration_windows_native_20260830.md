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

初始完整 pytest 结果：`1010 passed, 9 failed, 3 deselected, 25 subtests passed`，用时约 3 分 54 秒。9 个失败全部是测试直接读取的历史 WSL 输出未展开：1 个 v1 teacher checkpoint，以及 8 个 20260827/20260828 v2 的历史 training/audit JSON。对应内容已在 `outputs-manifest.jsonl` 和独立 tar 中完成 SHA-256 校验；按本次迁移方案不将这些历史文件展开到 Windows 工作树。它们现带有 `historical_artifact` pytest marker，默认回归套件排除该 marker；只有明确临时恢复命名归档后才执行该历史通道。当前代码、依赖、CUDA 和 smoke 运行验证均通过。

收尾复验结果：默认 pytest 为 `1010 passed, 12 deselected, 25 subtests passed`，退出码为 0；`historical_artifact` collect-only 精确收集 9 项。`compileall`、Ruff、v2 CLI、受影响测试模块、归档抽样 SHA-256 与 tar 目录读取均通过。Git bundle 含 66 个 refs 且完整历史可验证；五项核心备份的大小与 SHA-256 再次匹配。`git fsck --full` 未发现损坏；保留其可恢复 dangling objects，不在回滚期执行清理。

## WSL 注销收尾

Windows 验证、GPT App 历史会话人工验收和恢复点复核完成后，用户于 2026-08-30 明确授权注销 Ubuntu。最后检查确认注册名唯一为 `Ubuntu`、BasePath 为 `\\?\D:\WSL\Ubuntu`、无 Python/训练/WSL host 进程；随后仅执行 `wsl.exe --unregister Ubuntu`，命令退出码为 0。未注销 `docker-desktop`，未手动删除 `ext4.vhdx`。

注销后独立核验确认：WSL 注册表和 `wsl --list --verbose` 中不再有 Ubuntu，`D:\WSL\Ubuntu\ext4.vhdx` 及其目录已消失；`docker-desktop` 保留且停止。D: 可用空间由约 14.55 GiB 增至约 263.72 GiB。原始 WSL 会话、用户配置、增量数据、Git bundle 和 163 个 outputs tar 均继续保存在归档位置；未来 Linux 恢复只能通过重装 WSL 并恢复这些材料。

## 最终注销验收（2026-08-30）

- 当前分支 `research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1` 的 HEAD
  `75d0584c379b145b0f0eb45ece14fda160741f23` 已以普通方式推送到
  `https://github.com/jony-hash/codex-original.git`，远端 SHA 与本地一致。
- 最新恢复 bundle 为
  `C:\Users\24567\Documents\Codex\codex-original-final-20260830.bundle`，
  50,768,910 bytes，SHA-256 为
  `36F1C6E6FBEE03AB7DCB17B076895606F6166D193249248CB6FB9364512D57FE`；`git bundle verify`
  通过并包含当前 HEAD 与 55 个 refs。
- 注销后 Windows 环境复验：Python 3.11.16、PyTorch 2.5.1、CUDA 12.1、RTX 3060
  可用；v2 CLI、compileall、Ruff、迁移 verify、folder-bound verify、Sources verify
  通过；默认 pytest 为 `1,028 passed, 12 deselected, 25 subtests passed`。
- 注销后短 CUDA smoke 完成 2 个 one-shot step 和 epoch 验证，运行目录为
  `outputs/migration_smoke/20260830_191323_817115_train_windows_native_post_unregister`。
- GPT App 项目 `codex-original · Windows` 的主目录、辅助目录和导入会话已由用户人工验收；
  217 个唯一会话、147 条可见主聊天、70 条归档内部线程和 214 条 folder-bound 记录保持不变。

## Codex 会话迁移完成

2026-08-30 对 `wsl-codex-history.tar` 完成只读审计：218 个 JSONL 文件对应 217 个唯一 session ID，其中 1 个 normal/archived 重复项 SHA-256 完全一致；旧 `state_5.sqlite` 有 217 条线程和 10 条父子边，线程 schema 与 Windows 当前版本兼容。147 条 `source=vscode` 主聊天中，144 条记录旧 `codex-original` 工作目录，另外 3 条为杂项；70 条内部线程计划完整保留并默认归档。当前 Windows `.codex` 有 111 条线程，ID 与旧归档无重叠。

`scripts/maintenance/codex_history_migration.py` 已通过 compileall、Ruff、dry-run、canary 和 complete。它未覆盖认证或配置；先在 C 盘快照当前 sessions/SQLite，再以 canary→App 抽查→complete 的两阶段方式追加索引。最终迁入 217 个唯一会话（147 可见主聊天、70 归档内部线程），SQLite 完整性和 217 个 JSONL 哈希均通过；旧 10 条父子关系也已保留。脱敏知识层位于 `C:\Users\24567\Documents\Codex\codex-original-history-context`，不提交 GitHub。

## Windows 原生工作流与双项目桥接治理（2026-08-30）

本地执行入口固定为 GPT App 本地项目 `codex-original · Windows`：主目录是
`D:\codex-original`，辅助目录是 `C:\Users\24567\Documents\Codex\codex-original-history-context`。
云端规划入口为 `codex-original · ChatGPT`；它通过六文件 Sources 快照和标准交接块与本地
项目衔接。VS Code 仅打开同一 `D:\codex-original` 用于查看或手工编辑，不再打开已注销的
WSL 工作区。

当前 Windows 运行环境唯一是 `D:\ANACONDA\envs\optical-generative-models-win`。
旧 `D:\ANACONDA\envs\optical-generative-models` 保留为 Legacy，不删除、不用于当前
测试或训练；未激活终端中的裸 `python`/`pytest` 可能解析到系统 Python，因此操作手册
要求使用当前环境的显式解释器。

外部历史入口已从 `PROJECT_MEMORY.md` 改为 `HISTORY_CONTEXT.md`，仓库权威入口仍为
`docs\PROJECT_MEMORY.md`。Sources 生成器现在读取新的历史文件名；活动桥接输出固定为
`C:\Users\24567\Documents\Codex\codex-original-chatgpt-bridge\current`，支持
`refresh`/`status`，且只有 `status=current` 的快照才允许人工上传。ACL 异常的旧 Sources 派生目录已移入
`C:\Users\24567\Documents\Codex\codex-original-closure-quarantine-20260830`，
`.pytest_cache` 已先移入 `D:\codex-original-acl-quarantine-20260830\.pytest_cache` 后
重新生成；它们不进入项目主目录或当前 Sources 来源。默认不上传内部 manifest、原始
transcript、全部历史摘要、`.codex` 数据、outputs 或临时文件。
