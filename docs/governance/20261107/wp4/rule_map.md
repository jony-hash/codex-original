# WP4 规则映射（无规则丢失）

旧版 `AGENTS.md`（312 行）压缩为 165 行。逐条去向：

| 旧内容 | 新位置 |
| --- | --- |
| Claude Code / 主合同声明 | 文件头 |
| Current LightGen Research Policy（时间上限、digest、单判定、人工复审） | §3 不可协商项；§4.8（计划长度与写集） |
| GPU 显存读数 / CUDA 失败语义 | §3 第三条 |
| Scope Router | §1 |
| Execution Rules 1–8 | §4 第 1–8 条（逐条保留，措辞压缩） |
| Literature Quality Rule（5 条） | §5 Research Pointers + 新增 [literature_quality.md](../operations/literature_quality.md)（全文保留） |
| Sol / Luna 协作与复审路由 | §10 模型路由 |
| Codex / DSH 会话分工与并发写入 | §2 |
| Turn Opening Compliance | §1 末段 |
| Long-Training Gate | §5 Research Pointers + 既有 [long-training-preflight.md](../operations/long-training-preflight.md) |
| Ordinary Research Fast Path | §5 Research Pointers |
| Minimal Implementation And Failure Semantics | §5 首段 |
| Skill Policy | §10 首段 |
| Project Contract（v1/v2、光学链、field 语义） | §6 |
| Repository And Artifacts | §7 |
| Commands And Verification | §8（更新为 `run_windows_cpu_checks.py` / `--full`） |
| Boundaries | §9 |
| Capacity-map Research Lifecycle | §5 Research Pointers |
| Windows-native migration | §10 第三段（路径更新为 `docs/operations/`） |
| Fixed cloud execution / Codex network permissions | §10 第三段（压缩，细节留在 [autodl_workflow.md](../operations/autodl_workflow.md)） |
| Codex history / project memory | §10 末段（路径更新为 `docs/operations/`） |

## 有意的取舍

- 文献质量细则移到 `docs/operations/literature_quality.md`，根文件只留触发条件与指针；该规则在文献类任务中仍为强制。
- 云端执行与迁移的历史叙述压缩为"当前政策 + 指针"，历史日期不再写入根契约。
- 未新增 Gate、未改变任何科学判定；只改契约结构与路径。

## 与目标的差异

计划目标为 ≤150 行，实际 165 行。差异来自保留 8 条执行规则与 5 条不可协商项的完整语义；未删规则以凑行数。