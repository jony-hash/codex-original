# 当前项目状态

更新时间：2026-08-30

## 当前主工作区

- 主项目：`D:\codex-original`
- 基线分支：`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`
- 基线提交：`019b76ef911e0bd16b8d5d4169807323427684af`
- Windows 原生迁移提交：`6f9015b1ebc8a461f48801c5280ad13a168c14fd`（本地保留，回滚期后再推送）
- Codex 历史迁移工具检查点：`c383f21`（本地保留，回滚期后再推送）
- `origin`：`https://github.com/jony-hash/codex-original.git`
- Windows 环境：`D:\ANACONDA\envs\optical-generative-models-win`
- 历史输出归档：`D:\LDM-Archive\codex-original\outputs`

Windows 原生项目是当前唯一开发和训练入口。PowerShell 7、Windows Conda、CUDA 和 ChatGPT 桌面版 Codex 均指向上述项目；VS Code 只用于可选的代码查看和编辑。

## 迁移状态

- Git bundle、WSL Codex 会话归档、WSL 用户配置归档和 Conda 清单已保存到 `D:\LDM-Archive\codex-original\backup`。
- `outputs` 的 163 个一级目录均已独立归档并验证；manifest 记录 33,117 个文件、240,556,414,616 字节源数据和每个 tar 的 SHA-256。
- 归档保留 Linux 原始相对路径，包括尾随句点目录和长路径；历史输出不展开到工作树。
- WSL 的 `data` 已复制到 Windows 项目，10 个文件、121,987,494 字节，逐文件校验一致。
- 新环境已验证 Python 3.11.16、PyTorch 2.5.1、torchvision 0.20.1、CUDA 12.1 和 RTX 3060 Laptop GPU。
- Windows 原生 CUDA smoke 已完成训练和验证，结果位于 `D:\codex-original\outputs\migration_smoke\20260830_052257_572717_train_windows_native`。
- 初始完整 pytest：1,010 passed、9 failed、3 deselected、25 subtests passed；9 项均依赖按归档策略未展开的历史 WSL 输出，不属于当前 Windows 运行环境失败。
- 默认 pytest 现排除 `historical_artifact` 与 `scheduled_diagnostic`；历史通道保留 9 项测试，仅在明确临时恢复命名归档后运行。
- 收尾复验：默认 pytest 为 1,010 passed、12 deselected、25 subtests passed（0 失败）；`historical_artifact` collect-only 精确收集 9 项。
- Codex 历史会话已于 2026-08-30 完成 canary 与全量合并：旧归档 218 个文件中的 217 个唯一会话均已迁入 Windows Codex Home；147 条主聊天保持可见（144 条 `codex-original` 聊天的工作目录为 `D:\codex-original`，3 条杂项聊天保留原目录），70 条内部线程默认归档。
- 全量会话校验通过：SQLite `integrity_check=ok`、217/217 JSONL 文件与清单哈希可读、原有 Windows 会话未被覆盖；10 条旧父子关系已追加（迁移前 6 条，迁移后 16 条）。迁移前快照和不可变 WSL tar 均保留。
- 脱敏跨聊天知识层已生成到 `C:\Users\24567\Documents\Codex\codex-original-history-context`，包含 `PROJECT_MEMORY.md`、`CHAT_INDEX.md`、`CHAT_CATALOG.jsonl` 和 147 份摘要；常见密钥模式扫描为 0。
- 归档自检：163/163 个 tar 存在；抽查 4 个 tar 的 SHA-256 与目录读取通过；5 项核心备份大小与 SHA-256 均匹配。
- 最近容量检查：D: 可用约 14.59 GiB；回滚窗口内仅运行测试和小型 smoke。

## Windows 兼容性改动

迁移所需改动仅处理 Windows 运行环境、路径表示和历史证据兼容性，不改变模型结构、公共 API 或训练算法：

- 为 Windows 提供进程峰值内存统计兼容实现。
- 将实验产物中的相对路径统一序列化为 POSIX 形式。
- 将 Windows 下的临时二进制文件以可 `fsync` 的读写方式打开。
- 允许历史 WSL 数据集 manifest 在数据根目录迁移后保持原有证据身份，同时继续严格校验张量和 split 哈希。

## 回滚状态

WSL Ubuntu 与 docker-desktop 当前均为 Stopped，未发现 VS Code WSL 后台连接或训练进程。`vmmemWSL` 仅保留低于 1 GiB 的系统服务占用，不强制终止。迁移日为 2026-08-30，回滚窗口截至 2026-09-06；窗口内不在 WSL 项目上开发、训练或生成新输出，且 D: 不得低于 10 GiB。迁移提交保持本地，不在窗口内推送。Ubuntu 注销未执行，必须在窗口结束后再次取得明确批准。

## 下一步

以 Windows 项目作为唯一最新工作树；每次任务更新本文件中的提交、测试、实验路径和下一步。下一步是按 `docs/codex_history_migration.md` 手动完成 canary、App 验收和全量合并；2026-09-06 重新验收，通过后先推送本地迁移提交，再单独请求 Ubuntu 注销批准。
