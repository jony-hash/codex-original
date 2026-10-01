# ChatGPT 与 Codex 本地项目交接

更新时间：2026-10-01

ChatGPT 项目用于讨论、检索和计划；本地 Codex 项目 `D:\codex-original` 是源码、数据、Git、测试和科学结果的唯一权威。云端 Sources 是人工生成的只读快照，不提供本机访问，也不代替本地核验。

## 权威顺序

1. 根 `AGENTS.md` 和适用的 scoped 指令；
2. 本地 Git、源码、配置、数据、模型和正式结果；
3. `docs/CURRENT_STATUS.md` 与 `docs/status/current_handoff.md`；
4. `docs/PROJECT_MEMORY.md`；
5. 指定的历史归档；
6. ChatGPT 对话和 Sources。

Sources 与本地信息不一致时使用本地信息。云端报告的本地执行结果，在本地检查前标记为未核验。

## 计划与执行

交接只写一个研究问题、一个最终 Gate、明确输入、允许动作、固定科学边界和停止点。短计划优先；不要把路径清单、内部检查、修复次数或源码摘要扩展成额外的 Gate。

项目执行不设墙钟上限；时间和资源数据只作记录。研究任务不以源码、dirty patch 或文件摘要相等作为准入、续跑条件。执行前按明确路径核对实际模型、配置、checkpoint 步数、优化器、样本行、数据角色和冻结范围。保持数值、恢复和硬件安全。普通工程故障在授权范围内修复并验证受影响路径。

`MODE=IMPLEMENT` 只授权提示中明确列出的动作。它不自动授权新实验、增加更新、改变科学阈值、正式测试、promotion、云端训练或其他后继任务。

`INVALID_HARNESS` 不是科学负结果；只有完整有效的测量可以判为 `VALID_FAILED`。结果只更新本轮被授权的状态文件，不自动把历史摘要复制到多个文档。

推荐交接字段和反馈格式见 [`operations/delta_one_gate_short_prompt.md`](operations/delta_one_gate_short_prompt.md)。

## Sources 文件

当前六文件 bridge 输出名保持：

- `PROJECT_MEMORY.md`
- `CURRENT_STATUS.md`
- `CURRENT_HANDOFF.md`
- `V2_RUNTIME.md`
- `HISTORY_CONTEXT.md`
- `CHAT_INDEX.md`

canonical 本地来源为 `docs/PROJECT_MEMORY.md`、`docs/CURRENT_STATUS.md`、`docs/status/current_handoff.md`、`docs/architecture/v2_runtime.md`。历史上下文与聊天目录位于外部 history-context 文件夹，是导航材料，不是当前事实源。

刷新和上传仍由用户控制。先在本地运行 bridge 工具的 `refresh` 和 `status`，人工检查它生成的六个文件属于同一快照后再上传。只上传上述六个文件；不上传 manifest、原始 transcript、Git 元数据、outputs 或临时文件。工具不自动写入云端。

## Windows 本地环境

源码和测试在 `D:\codex-original`。支持环境解释器：

```text
D:\ANACONDA\envs\optical-generative-models-win\python.exe
```

Ubuntu WSL 已注销；`docker-desktop` 不是项目运行环境。不要使用旧 WSL 路径作为当前工作区。保留用户已有的 dirty 改动和实验产物。

## 停止与交接

完成授权范围后停止，不从旧计划自动创建新任务。若缺少必需输入、数据角色不清或真实硬件不安全，说明最小解阻项；普通可修复代码错误留在当前任务内修复。历史 evidence 保持原样，当前状态只记最新结论和一个后续建议。
