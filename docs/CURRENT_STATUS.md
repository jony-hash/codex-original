# 当前项目状态

更新时间：2026-08-30

## 当前主工作区

- 主项目：`D:\codex-original`
- 基线分支：`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`
- 基线提交：`019b76ef911e0bd16b8d5d4169807323427684af`
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
- 完整 pytest：1,010 passed、9 failed、3 deselected、25 subtests passed；9 个失败均为按归档策略未展开的历史 WSL 输出读取失败，已在迁移记录中列出，不属于当前 Windows 运行环境失败。

## Windows 兼容性改动

迁移所需改动仅处理 Windows 运行环境、路径表示和历史证据兼容性，不改变模型结构、公共 API 或训练算法：

- 为 Windows 提供进程峰值内存统计兼容实现。
- 将实验产物中的相对路径统一序列化为 POSIX 形式。
- 将 Windows 下的临时二进制文件以可 `fsync` 的读写方式打开。
- 允许历史 WSL 数据集 manifest 在数据根目录迁移后保持原有证据身份，同时继续严格校验张量和 split 哈希。

## 回滚状态

WSL Ubuntu 保留为只读回滚源，迁移日为 2026-08-30，计划回滚窗口截至 2026-09-06。已执行停止，但现有 VS Code WSL 连接会自动重新启动 Ubuntu；关闭该 WSL 窗口后再执行一次停止即可释放其虚拟机。窗口内不在 WSL 项目上开发、训练或生成新输出。Ubuntu 注销未执行，必须在回滚窗口结束后再次取得明确批准。

## 下一步

关闭现有 VS Code WSL 窗口并再次确认 Ubuntu 为 Stopped 后，以 Windows 项目作为唯一最新工作树；每次任务更新本文件中的提交、测试、实验路径和下一步。
