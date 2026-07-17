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
- `ldm_project` 根包的模型导出采用惰性属性；data-only/student stage 不应因兼容 root import 加载 Teacher 网络。D2NN tensor/metric primitive 位于中立 `evaluation`，`student` 不得导入 `diagnostics`。
- `configuration/schema.py`、`configuration/loading.py`、`configuration/validation.py` 分别拥有 canonical config 实现；`configuration/legacy.py` 与根 `config.py` 不得重新承载算法主体。`teacher/export.py`、`label_audit.py`、`filter.py`、`readable.py` 拥有 canonical distill workflow，`teacher/common.py`、`distill.py` 和 legacy facade 只做转发/兼容注入。实验 objective term 位于 `student/objective_terms/`，`student/objectives.py` 只做编排。
- Formal v56 stage 使用 `FormalOneShotOpticalGenerator`；`student.generator`、`student.checkpoints`、`optics.d2nn` 和 `distillation.targets` 分别拥有 generator、checkpoint、phase-only D2NN 和中立 target transform 合同。`student.experimental` 承载兼容诊断层，`student.layers` 仅为旧 imports 提供 shim；历史 `OneShotOpticalGenerator` 仅用于 compatibility/diagnostic 配置。
- `experiments/archive/` 只读、`diagnostic-only`，不能作为 active config/tool promotion。
- v68 仅允许 `configs/train_v68_transparent_readability_a.yaml`（camera-only 上界）和 `..._b.yaml`（A 通过后的最小可读性分支）作为透明诊断入口；先运行 LightGen 合同审计与数值/显存准入，再由用户执行长 CUDA。透明 `checkpointed_active_capture` 只减少保存/反向激活，不改变完整复场、backend relay 或 Formal v56/v60。
- v68.3 的 raw-CMOS 硬门必须使用 `tools/audit_v68_raw_cmos.py` 的一个全 validation 共享 exposure scale、gamma=1 和 `physical_camera_intensity_raw`；历史逐样本增强图只作对照。A checkpoint 若缺少显式边界字段，只能由其 train config 推断为 diagnostic transparent-v2，Formal loader 必须拒绝。A2 配置的固定曝光 scale 只能来自该审计，未通过人工 raw 门不得启动 B。
