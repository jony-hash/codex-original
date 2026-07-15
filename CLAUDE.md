# CLAUDE.md

- In the first assistant message of each new task, the first line must be exactly `【CLAUDE.md effective】`.
- Do not repeat this marker later unless the user explicitly asks for it.

## Claude Code 行为规则

开始任务时先读取仓库根 `AGENTS.md`；修改 scoped path 或执行仓库验证前，按其中的 Scope Router 读取适用的 scoped `AGENTS.md`。项目架构、光学语义、验收、checkpoint、输出政策和受保护接口均以这些合同为准，本文件不得重定义它们。

1. 改代码前读取必要文件；不确定时查询本地事实，不猜接口或业务。
2. 仅修改请求范围，遵循既有命名、结构、类型和风格；关键物理边界可使用简洁中文注释（SLM command、sampling/FOV、relay ABCD、D2NN padding/propagation/crop、CMOS 与 checkpoint 兼容）。
3. 新增或修改行为前读取调用者和被调用者，优先复用既有实现。
4. 仅在实际执行对应验证后报告成功；准确说明跳过或无法验证的部分。
5. 在 `AGENTS.md` 授权范围内直接执行代码修改、测试、lint、compileall 和 web 搜索；需要提权、删除文件、force push 或 AGENTS.md 指定的受保护改动时先询问。
6. 当两个合理解释会导致不同结果、用户目标与字面要求冲突，或需要密钥、凭证、硬件访问或不可达外部资源时，先说明事实并请求方向。

## 关键目录

- `optics/`、`student/`、`teacher/`、`application/` 是正式领域；顶层旧模块仅为无 warning 的 import compatibility shim。
- `application.stages.StageRequest` 按 stage 懒构造依赖；不得恢复 CLI 启动时同时实例化 VAE、DDPM、scheduler 和 student 的模式。
- Formal v56 stage 使用 `FormalOneShotOpticalGenerator`；`student.checkpoints`、`optics.d2nn` 和 `distillation.targets` 分别拥有 checkpoint、phase-only D2NN 和中立 target transform 合同。历史 `OneShotOpticalGenerator` 仅用于 compatibility/diagnostic 配置。
- `experiments/archive/` 只读、`diagnostic-only`，不能作为 active config/tool promotion。
