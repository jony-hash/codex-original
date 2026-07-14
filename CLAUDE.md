# CLAUDE.md

- In the first assistant message of each new task, the first line must be exactly `【CLAUDE.md effective】`.
- Do not repeat this marker later unless the user explicitly asks for it.

## Claude Code 行为规则

开始任务时先读取仓库根 `AGENTS.md`；修改 scoped path 或执行仓库验证前，按其中的 Scope Router 读取适用的 scoped `AGENTS.md`。项目架构、光学语义、验收、checkpoint、输出政策和受保护接口均以这些合同为准，本文件不得重定义它们。

1. 改代码前读取必要文件；不确定时查询本地事实，不猜接口或业务。
2. 仅修改请求范围，遵循既有命名、结构、类型和风格；不顺手重构、加注释或改格式。
3. 新增或修改行为前读取调用者和被调用者，优先复用既有实现。
4. 仅在实际执行对应验证后报告成功；准确说明跳过或无法验证的部分。
5. 在 `AGENTS.md` 授权范围内直接执行代码修改、测试、lint、compileall 和 web 搜索；需要提权、删除文件、force push 或 AGENTS.md 指定的受保护改动时先询问。
6. 当两个合理解释会导致不同结果、用户目标与字面要求冲突，或需要密钥、凭证、硬件访问或不可达外部资源时，先说明事实并请求方向。
