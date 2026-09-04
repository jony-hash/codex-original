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

## WSL 历史 outputs 恢复（2026-08-31）

用户要求将原 WSL 项目的历史 outputs 重新显示在当前 Windows 项目中。本次使用
`D:\codex-original\outputs\_wsl_restore_metadata_v4\restore-plan.json`，先完成
`archive_0001`、`archive_0011`、`archive_0158`、`archive_0002` 四个 canary，再按断点完成其余
归档。最终结果为 163/163 个归档、33117 个历史文件、240556414616 个源字节；完整 verify
通过，D: 可用空间约 40.57 GB。

恢复仅补入缺失的历史路径并对 28 个重叠文件逐字节校验；相同文件保留现有 Windows 文件，
不同内容会 fail-closed，实际未发生覆盖。Linux 无法在 NTFS 中直接表达的 32 个尾随点路径和
14 个超长路径已确定性映射，原始 Linux 路径与 Windows 路径记录在
`outputs\_wsl_restore_metadata_v4\path-map.jsonl`。原始 163 个 tar、index、manifest 和旧
Codex/WSL 归档均未修改。

恢复工具 `scripts/maintenance/restore_wsl_outputs.py` 的 9 项定向测试、compileall、Ruff、
v2 CLI 和正确 Windows CUDA/PATH 条件下的完整 pytest 均通过：`1055 passed, 12 deselected,
25 subtests passed`。历史 marker 通道在可写临时目录下为 5 通过、4 失败；失败原因是历史 JSON
保留 WSL checkpoint 绝对路径，和 Windows 当前解析路径不一致。该历史证据保持原样，未将其
转换为当前科研结论，也未修改模型、训练代码或公共 API。

恢复后 `D:\codex-original\outputs` 是“当前 Windows 结果 + 历史 outputs”的可见目录；原始 tar
仍是不可变恢复点。权威 v4 恢复状态和路径映射仍在项目 metadata 目录；旧 v1–v3 元数据与
失败 staging 已移动到 `D:\LDM-Archive\codex-original\reports\restore-attempts-20260831`，
空 staging 根目录已删除。三个 pytest 临时目录因当前身份 ACL 仍未删除，明确视为测试残留，
不进入 Git；恢复 outputs、归档和历史证据也未进入 Git。

## 迁移后当前研究核验注记（2026-09-01）

Windows 原生主工作区当前在分支
`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`，RMAL implementation commit 为
`5b274b926474784a1ad49744e75190afff31747c`。本次只新增 RMAL producer、配置、review contract、
测试和 research CLI 路由，未修改正式 v2 runtime、OLS、checkpoint、训练目标、detector、prior
或历史 evidence。

RMAL 的唯一权威 CUDA attempt-01 为
`D:\codex-original\outputs\20260901_v2\runs\lightgen_scaled_chen_reconstruction_mapping_alignment_localization_v1_attempt_01\`；validation/audit/compact/probes SHA-256 为
`affb4961626a2caaf4dc2c8bb04c06ac4f8de51d1d4a1ab418858eaa11e536ec`、
`5ba51e041a266865fc0443b729d31a0c97c35f2b86d6ef45f11dacf6617a15fa`、
`c335f75124666e9423d66b7a4537c74ff642ef2d71044bda86be16f9333a2137`、
`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`。
静态 validation 与正式 audit 均通过；首个下游 `cmos_intensity` 在 selection/confirmation 均通过，
其余上游阶段按 stop rule 跳过。唯一下一动作是
`manual_review_RMAL_recoverability_result`；不授权训练、FMF、prior sampling、generation 或
formal promotion。RMAL 定向测试 14 项、直接相关回归 56 项、compileall/Ruff/diff-check 均通过；
全套 pytest 的既有 near-conjugate 临时路径失败仍单独记录，未修改无关代码。

## H-LOMA-CBOGT-S5-ZCSA（2026-09-01）

Windows-native 环境完成 ZCSA zero-column semantics audit。起始 HEAD 为
`0dead1b73655b3556450cedbb5d0a2c579385209`，实现提交为
`1f5e553f6c7ad80217427ab83f5669e7b9758acb`；新增 producer/config/review/tests/CLI 路由，未改动
RMAL/RWSA 旧产物、probe、renderer 或 formal runtime。正式 attempt 为
`outputs/20260901_v2/runs/lightgen_scaled_chen_detector_readout_zero_column_semantics_audit_v1_attempt_01/`，
validation/audit/compact SHA256 为 `d422f1e31e8298f9662896f832b207ad896fb619c5a656bccf54b099cd419954`、
`0cba5990462e9f8cc238885b14c90c907a7c6f085877b6c8ad0c8b95cbf0bfb8`、
`772fb739b39d029c213ab42e8f552648782607b5562b6350d4d2c9744b73b9e6`。

审计通过，classification=`zero_column_semantics_attribution_completed`；dynamic active/zero-W 为
`33502/6498`，zero-W 全部为 `target_zero_static`，其余 taxonomy 类别为 0。fit-zero→selection/
confirmation active=`1495/2097`；b exact-zero=`6498`；zero mask pixel/edge/interior=`6498/660/5838`，
4-connected components=`2`（最大/次大=`5105/1393`）。无 optical replay、CUDA forward、model load、
optimizer、backward、evaluator 或 checkpoint 写入。定向测试 18 项、直接相关回归 57 项、compileall、

## H-LOMA-CBOGT-S5-RWSA-ACTIVE：Detector Readout Structure Attribution（Active-Domain Amendment，2026-09-01）

本次是用户批准的 RWSA active-domain amendment。真实消费者是项目负责人判断冻结 RMAL detector readout 在 dynamic-active 输出域上的空间结构、符号与 bias attribution；拒绝决定是不能把 descriptive structural result 写成 universal representation sufficiency、架构修正、reconstruction/generation readiness 或 training/promotion 授权。

起始 lineage HEAD 为 86e0862690cdb32173e453869ed33b8eb2b4c017。实现与修补依次落在 eed3bb3、6521893、ca372c9、e0c442f；正式 validation/audit 的 clean implementation/result commit 为 e0c442f92dd23b71d048ceb647570b76406c0435。新增 active-domain producer、研究 YAML、review contract、测试和 CLI schema/router；旧 RWSA attempt-02、ZCSA/RMAL 产物、冻结 probe、target renderer、checkpoint、OLS、detector、formal runtime 与历史 evidence 均未修改。

因 write-once 规则，attempt-01 只保留 validation-only retry 记录：目录为 outputs/20260901_v2/runs/lightgen_scaled_chen_detector_readout_structure_attribution_active_domain_v1_attempt_01/，validation SHA-256 为 8c20e5629e36e2d88d2358512b95f5cd0858999e21d981f864ea8f541f44a539；run 在 transition harness 修复前停止，未写 audit，不消费该 attempt。权威 attempt-02 为 outputs/20260901_v2/runs/lightgen_scaled_chen_detector_readout_structure_attribution_active_domain_v1_attempt_02/，validation/audit/compact SHA-256 分别为 686d99f331c99b30a73cb3a70cc1002aa0a21a33b014e2a59a5e6c33f8a5e15d、cb3feabbfc0b70e23d43ae18a354dabc112319eb11d2c8cea2c13f6bd5c0337、23b38d3231594cb0eb3ebab21972b8979ffaf825e17c901f4b86d46eeddc1c11；active config canonical SHA-256 为 2115380c676fc710470429abed9438d154a1758db99e5635d6295eeaae03a25d。

冻结 probe 仍为 RMAL 的 [257, 40000] float32 full readout（W=[256,40000]、b=[40000]），probe SHA-256 为 3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed。dynamic mass 定义为 d_j=sum_i abs(W_ij)，active 仅取 d_j>0，inactive 取 d_j==0；完整域为 40000，dynamic-active 为 33502，zero-W 为 6498，W 与 b 在 inactive 列均 exact zero，active mask SHA-256 为 afbda35eca136c1275c5771bc9540f7985dd66f77a4c6ea596940649848352ea。inactive 概率不计算、不用 epsilon、不静默删除，所有 inactive 列保留在 decomposition。

fit 使用预注册 1000 个 train fit 样本（每类 100），只定义 RMAL/ZCSA identity；selection 是 MNIST test 5000–9999 的 1000 个样本，confirmation 是 train 55000–59999 的 5000 个样本，未重拟合 readout，candidate prefix 未消费。target 仍由固定 base.render_intensity_batch 产生；16×16 adaptive-average-pool 使用重叠 bins (0,13)、(12,25)、…、(187,200)，坐标为 centered row=y、column=x、pixel pitch 2.5e-6 m、FOV 0.5 mm。D4 controls 为 identity、x_flip、y_flip、xy_flip、transpose、transpose_x_flip、transpose_y_flip、transpose_xy_flip。permutation null 为 source-feature-row-only、同一 active mask、256 次、seed=0，仅描述不设 scientific threshold。

正式结果为 status=passed、classification=descriptive_structural_attribution_completed，且 conditional_on_dynamic_active_domain=true。L_active=0.003260666999；top-1/top-4 mass=0.0806218534/0.2238490484，expected-support locality=0.0032606670，radius 0/1/2 locality=0.0032606670/0.0241852561/0.0619379424；spatial variance=7.1791287e-08 m²，RMS support radius=267.8815 µm，effective support=44.3347，normalized entropy=0.8016757，row/column concentration=0.2319700/0.2541995，positive/negative mass=0.4990808/0.5009192，cancellation fraction=0.9248245。null mean/std=0.0042417098/0.0003771318，observed empirical rank=1/256；这些数值是 descriptive structural statistics，不构成 bottleneck 或 readiness 判定。

两组 exact CMOS-only no-update replay 的 registered physical anchors 均通过：selection physical paired/template/cyclic MSE=0.0620484594/0.0324799946/0.0799509163，confirmation=0.0577895189/0.0277411913/0.0756082949，均在既有 tolerance 内。selection 的 all_40000 paired/full/dynamic/bias MSE=0.0620484594/0.0098129277/0.0126702857/0.0984088076，dynamic_active=0.0740748917/0.0117125174/0.0151240844/0.1174923265，zero_W=4.3298663e-05/1.9136666e-05/1.9136666e-05/1.9136666e-05。confirmation 对应为 all_40000=0.0577895189/0.0101852894/0.0130028048/0.0876878199，dynamic_active=0.0689895374/0.0121560417/0.0155200385/0.1046908519，zero_W=4.5133395e-05/2.4601896e-05/2.4601896e-05/2.4601896e-05。

zero-W error contribution（paired/full/dynamic-only/bias-only）为 selection=0.0001133609/0.0003168016/0.0002453576/0.0000315902，confirmation=0.0001268728/0.0003923873/0.0003073628/0.0000455773。fit-zero 与 zero-W mask 完全一致；fit-zero→selection target-ever-nonzero transition=1495 列，fit-zero→confirmation target-ever-nonzero transition=2097 列，transition error 仍只归入 zero_W_transition_columns。zero-W/active masked class-template、cyclic 和 instance-sensitive metrics 按合同均为 not_evaluated_without_unambiguous_masked_output_semantics；未保存 raw CMOS bank 或大 feature bank。

回放只执行 exact panel CMOS model replay：compact 记录 exact_panel_cmos_only_replay_executed=true、optical_replay_executed=false；optimizer_created=false、backward_calls=0、evaluator_calls=0、checkpoint_written=false、prior_sampled=false、generation_started=false、physical_forward_modified=false，模型/OLS/checkpoint/artifact integrity 全部保持不变。资源峰值 CUDA=303070720 bytes，wall time=183.036 s，最终三文件总量=240004 bytes，resource/storage gates 通过。

全套默认 pytest 在显式短 basetemp 下为 1214 passed、12 deselected、25 subtests passed；新增 active-domain 定向测试为 17 passed，直接相关回归为 96 passed；compileall、Ruff、git diff --check 与 CLI contract checks 均通过。唯一 warning 是既有 .pytest_cache ACL（C:\TEMP / D:\codex-original\.pytest_cache），不是测试失败。

本结果仍保持 reconstruction_manifold_supported=false、generative_manifold_supported=false、generation_readiness=not_ready、training_authorized=false、formal_promotion=false 与 closed_task_relevant_bottleneck_not_supported。唯一下一动作是 manual_review_RWSA_active_domain_result；不得训练、FMF、prior sampling、generation、candidate export 或 formal promotion。

## H-LOMA-CBOGT-S6-LSDA：Linear Readout Structure Decomposition Audit（2026-09-02）

Windows-native 环境完成 LSDA frozen-W、CPU-only descriptive spectral/operator attribution。起始 HEAD 为
`bd967f64ffb046677fa4204e81e911b226047e6e`，实现 commit 为
`bed235283d1fa47f4a5a38d41368d87ce2b5d218`；新增 producer/config/review/tests/CLI schema-router，未修改 RMAL/ZCSA/RWSA-ACTIVE、probe、renderer、checkpoint、OLS、detector、formal runtime 或历史 evidence。

正式 attempt：
`outputs/20260901_v2/runs/lightgen_scaled_chen_linear_readout_structure_decomposition_audit_v1_attempt_01/`；validation/audit/compact SHA256 为
`3f24dcc7d56e098ec778998bb8ddc7871973131f1d7269c86da729395426c2f2`、
`97842d5707541516c79adb87c3a1e609ee533156895908c9e99c015cde0f9b13`、
`e281daf9a4a62f85828a3bc8055ce044027d0c0fb1af0a522febcf3c2b68185b`；probe SHA256 仍为
`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`。结果为
`passed / linear_readout_spectral_attribution_completed`。

W/b=`[256,40000]/[40000]`，full/active/inactive=`40000/33502/6498`，full/active CPU float64 Gram=`[256,256]`；前五 singular values=`12227.8671591/10625.3503547/8724.4504853/8126.3713707/7844.9489846`，top-16 cumulative=`0.8441783201`，effective/stable telemetry=`21.88110383398559/21.881103833985602` 与 `5.528997319328641/5.5289973193286395`。full-vs-active 最大 eig absolute/relative difference=`4.470348358154297e-08/8.328885261127632e-13`，sigma difference=`2.900790718740609e-12`，inactive identity 保持。

K=16 仅为 descriptive preview；source/target orthogonality residual=`1.5695833278775036e-15/3.095197802817193e-14`。global reference 为 RMAL fit panel 1000 个 unlabeled streaming targets，hash=`513a696e21dafebebe4a862ab814b06294beb662db2afcfd679899fd19b27daf`；class-template、selection、confirmation 与 target bank 未使用/保存。bias norm²=`114.5276114768678`，preview projection energy=`21.21861242530154`，preview 外 residual=`93.30899905156626`。

validation/audit gates 全部通过；全程无 CUDA、optimizer、backward、refit、decoder、physical forward 或 rank-truncated reconstruction。新增定向测试 `18 passed`，直接相关回归 `101 passed`，显式可写 basetemp 的完整 pytest 为 `1232 passed, 12 deselected, 25 subtests passed`；compileall、Ruff、diff-check 通过，唯一 warning 为既有 `.pytest_cache` ACL。状态 flags 保持 `reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false` 与 `closed_task_relevant_bottleneck_not_supported`；唯一下一动作是 `manual_review_LSDA_spectral_result`。

## H-LOMA-CBOGT-S6-LRRC：Low-Rank Readout Recovery Confirmation（2026-09-02）

Windows-native 环境完成 LRRC 的 frozen-W、no-refit functional recoverability confirmation。实现、
配置、review contract、测试与 CLI 的 clean commit 为 `2a061db61910a847118e38663d05c05bd9447063`；
RMAL/ZCSA/RWSA-ACTIVE/LSDA、probe、renderer、checkpoint、OLS、detector、formal runtime 与历史
evidence 未修改。权威 attempt-04 位于
`outputs/20260902_v2/runs/lightgen_scaled_chen_low_rank_readout_recovery_confirmation_v1_attempt_04/`；
validation/audit/compact SHA256=`b305c38dc3c774ff995c98907977d67cfc396d56009b8a88f726e214f9efa004` /
`5ffcb3550e0a1d635f70cdf965d12c93ded6e99fadf8f0f6ef0cda829b4a6b58` /
`4ae704104accd8cebfcb0e3c19dba7deb90b4febf38bace3c7e578bd484951b8`。

candidate family 固定为 `[1,2,4,8,16,32,64,128,256]`，rank-256 full positive anchor first 且
selection/confirmation 均通过；rank `1,2,4,8,16,32,64,128` 按预注册顺序全部 selection 失败，
confirmation 未执行。故 `first_confirmed_candidate_rank=256`、`rank_256_reused_as_fallback=true`，
没有第二次 rank-256 run。operator identity、CMOS-only replay、state/no-update、streaming/no-bank
和 resource gates 全部通过；`cmos_only_replay_executed=true`、`optical_replay_executed=false`。

结果为 `passed / low_rank_readout_recovery_confirmation_completed`；LSDA boundary gaps 与
rank-36 smallest relative gap=`0.0025793411` 仅作 descriptive telemetry，无新增 threshold。
flags 继续为 `reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false` 与
`closed_task_relevant_bottleneck_not_supported`。最终核验：LRRC `26 passed`、相关 `130 passed`、
默认全套 pytest `1258 passed, 12 deselected, 25 subtests passed`；唯一下一动作是
`manual_review_LRRC_result`，结果/状态 commit 后 refresh bridge 并检查 status，不自动上传。

## H-LOMA-CBOGT-S6-DSFA：Data-Weighted Spectral Functional Attribution（2026-09-02）

Windows-native 环境完成 DSFA frozen-RMAL、no-update data-weighted spectral functional attribution。
实现/config/review/test/CLI 的 clean commit 为
`d032738c0f02af5ada85554224e045d238416e0a`；上游 RMAL/ZCSA/RWSA-ACTIVE/LSDA/LRRC、probe、
renderer、checkpoint、OLS、detector、formal runtime 与历史 evidence 未修改。权威 attempt 为
`outputs/20260902_v2/runs/lightgen_scaled_chen_data_weighted_spectral_functional_attribution_v1_attempt_01/`，
validation/audit/compact SHA256 为
`ff15d6da857f1db12b67b2abb0a182aff0a622b15718c44113ad01389fe7efbe` /
`2521e0ec4138d55cfaf17f153a493c40aef5151bf5628a4daa9852dc198a265e` /
`8aec1e2b718eb6de499f87d81b68203ebbd7d8483bff0c36066ed9e376dea3c3`。

结果为 `passed / data_weighted_spectral_functional_attribution_completed`。selection/confirmation
full-probe MSE=`0.009812927922827133/0.010185289342506197`，B256 operator-energy share
`0.0007625489` 对应 data-weighted output-energy share `0.3517092/0.3451470`。LSDA q/operator、
两 panel additive identity、LRRC replay、state/no-update、streaming/no-bank、finite 和 resource
gates 全部通过；peak CUDA=`2274099712` bytes，wall=`225.532 s`。定向测试 `10 passed`，
compileall/Ruff/diff-check 通过；唯一下一动作是 `manual_review_DSFA_result`，不授权训练、
generation 或 promotion。

## H-LOMA-CBOGT-S6-DMSA：Detector Data-Metric Spectral Attribution（2026-09-02）

Windows-native 环境完成 DMSA frozen-RMAL、no-update detector data-metric spectral attribution。
起始 lineage HEAD 为 `39b0ac449986df31b9102c0b1296dda912f7b39c`；实现/config/review/test/CLI
的 clean implementation/result commit 为 `9818e87441eb2c86eef06c18c1aa646f3e57dea5`。
上游 RMAL/ZCSA/RWSA-ACTIVE/LSDA/LRRC/DSFA、probe、renderer、checkpoint、OLS、detector、
formal runtime 与历史 evidence 未修改。

attempt-01 为 validation-only invalid harness（`three_panel_isolation` 修复前停止），validation
SHA-256=`94a29d7ad6c065ac0253700524de1f0f1e23c0e92fe3959b340ae04e26ac63b3`，按 write-once
规则保留且不消费。权威 attempt-02 为
`outputs/20260902_v2/runs/lightgen_scaled_chen_detector_data_metric_spectral_attribution_v1_attempt_02/`；
validation/audit/compact SHA-256 为
`f0eebcb978851b9ab58453e1b28e920eddc93e86a2b82133af0bd7ec8e8ce93d` /
`c9cc448f1b58040d45b8b77c9cfaa9facd536cdcf40d0b4f152efe42e16411fe` /
`39564f0ce3973bd738d196191a06c7990407f9b7b6ab7e6e8a2eb3366dc2d0ac`；research config/review/source
SHA-256 为 `48e3e34d47a486fd50052f2d805d387c84fc216a9e9498775059c2de45560828` /
`2899715f961a19cdee83b53f709c9b77498ab618afe5963c6c63a5492ea2501a` /
`88356728b165e7c1ff91fb825292a635901da2d02c60841da972b6434d36112f`。

结果为 `passed / detector_data_metric_spectral_attribution_completed`。fit/selection/confirmation
使用 1000/1000/5000 样本与 population `N` streaming moments；15/15 audit gates、raw LSDA
identity、三组 moment identity、selection/confirmation cross identity、state/no-update、
no-bank、finite 与 resource gates 全部通过。selection/confirmation mean/variance energy
fractions=`0.4019555898/0.5980444102`、`0.3704005497/0.6295994503`；second-moment
effective/stable rank=`14.8728108/2.2436135`、`17.4264177/2.4233991`，centered-variance
为 `36.6563260/8.1270337`、`39.8650006/8.4787742`。peak CUDA=`2274099712` bytes，
wall=`84.922 s`，exact-panel CMOS-only replay=true，optical replay=false，optimizer/backward/
refit/decoder/generation 均为 false/zero。

定向测试 `21 passed`，直接相关回归 `111 passed, 1 deselected`，compileall、Ruff、diff-check
与 CLI checks 通过。全套 pytest 为 `1285 passed, 12 deselected, 25 subtests passed`，另有
4 个无关环境失败（3 个 NVRTC DLL/运行时失败、1 个 near-conjugate 临时路径失败）。状态
flags 继续为 `reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false` 与
`closed_task_relevant_bottleneck_not_supported`；唯一下一动作是 `manual_review_DMSA_result`，
不授权训练、decoder、candidate export、generation 或 promotion。

## H-LOMA-CBOGT-S6-MCFA（2026-09-02）

Windows-native 环境完成 MCFA frozen-RMAL、no-update detector mean-centered functional
attribution。实现/config/review/test/CLI 的最终 clean implementation/result commit 为
`069cfa4b3aeda3943d754d739daf6daeba2a54a6`；新增文件/路由仅限 MCFA producer、YAML、review
contract、tests、research CLI schema 与 lazy router，RMAL/ZCSA/RWSA-ACTIVE/LSDA/LRRC/DSFA/DMSA、
probe、renderer、checkpoint、OLS、detector、formal runtime 与历史 evidence 未修改。

attempt-01 为 validation-only failed harness，正式 replay 在 panel image runtime binding
处停止，validation SHA-256=`e27c273780387ceb36f1c63cb34746231356236d209ac122f61a2a23a35bb3bb`；
按 write-once 规则保留且不消费。权威 attempt-02 为
`outputs/20260902_v2/runs/lightgen_scaled_chen_detector_mean_centered_functional_attribution_v1_attempt_02/`；
validation/audit/compact SHA-256 为
`35ad479aa931cb2e605bb9b582306549512106a5c2dcd070bfde35a2e552e7f1` /
`f69664cf9da1d61e90742a09614c4535388402f9228e401d2451e3ebeeefdc0b` /
`e95c072bde8306dd3f7cb8d6c5af1a671a778cede8f84c285c450af854059494`；probe SHA-256 仍为
`3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed`。

结果为 `passed / detector_mean_centered_functional_attribution_completed`。fit/selection/
confirmation 样本为 `1000/1000/5000`，两遍 CMOS-only streaming replay 共 14000 visits；
mean/centered output-energy fractions=`0.3829179441/0.6170820559`、`0.4019555898/0.5980444102`、
`0.3704005497/0.6295994503`，mean/centered/total functional gains=`0.0313323622/0.0504929026/0.0818252648`、
`0.0396463978/0.0489494817/0.0885958795`、`0.0316138044/0.0458887261/0.0775025305`。
15/15 audit gates、cross-identity、state/no-update、no-bank、finite、attribution boundary 与
resource gates 全部通过；peak CUDA=`2274099712` bytes、wall=`183.266 s`。完整 pytest 为
`1310 passed, 12 deselected, 25 subtests passed`，compileall、Ruff、git diff-check 通过。
治理 flags 保持 `reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false` 与
`closed_task_relevant_bottleneck_not_supported`；唯一下一动作是 `manual_review_MCFA_result`，
不授权训练、decoder、candidate export、generation 或 promotion。

## H-LOMA-CBOGT-S8-MPCDAR-R16-IQ（2026-09-02）

Windows-native 环境完成 S7 MPCDAR rank-16 frozen artifact/runtime integration qualification。
实现 clean commit=`c0140577fddb56af79d72a6d473cc49ef04f5ffc`；权威 attempt-03 为
`outputs/20260902_v2/runs/lightgen_scaled_chen_mpcdar_rank16_integration_qualification_v1_attempt_03/`。
validation/audit/compact SHA256=`844a9dc4187623c380655db63530e13e6ba1a9d4e62d6ef44dd95b29119457cb` /
`3b98c0faa6816b11d2f6da8cc832e89d221344f4b01bd41eadb356968f1b39a4` /
`12e342d75119ee143186046490bb14c064f449dc27520168a17e84ae758b8271`；artifact manifest/payload
SHA256=`e580dd59f3e277055a9c0cff6d9f076e06346b5761a8b9f228608e88b3a23c45` /
`e839daaff8d718347d1cd4194651ac0773f842ffa39fe4efaa3e0218ea2e1d4e`。旧 attempts 按
write-once 保留且不消费。

Process A build 后退出，Process B fresh reload 完成；四个 CPU float64 tensor 的 shapes 为
`[256]`, `[40000]`, `[256,16]`, `[40000,16]`，scalar count=`684352`，raw/serialized payload
bytes=`5474816/5476856`。mu/C fit replay hash 匹配，moment identity relative residual=
`2.98546e-16`；rank-256 positive replay max abs/relative-Frobenius residual=
`5.68008e-11/1.17382e-13`。selection/confirmation loaded/reference residuals 均为零；
probe MSE=`0.02107268348957069/0.02128368811373826`，relative improvement=
`0.6603834534521028/0.6317033169406975`，confirmation bootstrap 95% CI=
`[0.03618517890572548,0.036830976605415344]`；RMAL Gate vectors 与 S7 parent 相同且通过，
audit=`16/16`，classification=`frozen_mean_preserving_data_aware_rank16_integration_qualified`。

factorized/full-W deterministic operation-count telemetry=`644096/10240000` MAC/sample，ratio=
`15.9`，不是 latency/throughput/energy-efficiency 测量。runtime independence、default-disabled、
explicit-artifact、CMOS-only streaming/no-bank、state/no-update、finite/resource gates 通过；
peak CUDA=`2274099712` bytes，wall=`133 s`。状态保持 reconstruction/generative false、
generation_readiness=`not_ready`、training/promotion=false 与
`closed_task_relevant_bottleneck_not_supported`；唯一下一动作是
`manual_review_MPCDAR_R16_IQ_result`，不自动 make rank16 default、训练、decoder、generation
或 promotion。Bridge 仍需在 worktree clean 后按合同独立执行 refresh/status；不自动上传或修改
S8 evidence。

## H-LOMA-CBOGT-S9-MPCDAR-R16-F32-NQ（2026-09-02）

Windows-native 环境完成 S8 rank-16 frozen artifact 的独立 pure-float32 CPU numerical qualification。
实现/fix commits=`4b75e4d67efc1dc90a01d1d33a6aa3c1c243c652` /
`ebbd0d232c56ee45a69dea830911eff4f697ad47`；attempt-01 是 write-once 保留的 serialization
harness failure，权威 attempt-02 为
`outputs/20260902_v2/runs/lightgen_scaled_chen_mpcdar_rank16_float32_numerical_qualification_v1_attempt_02/`。
validation/audit/compact SHA256=`5728fb9b50212d5dc7359975b06fe6d3f51be3645a23410683334abab7bb2137` /
`0ff4aeaddd2035f0ced865a5e9f76331db53a2a8a213ea18064bd8e1d7368bde` /
`5b45f1401489fab63382366045a95f52984505038a379ca8bd8d0a780424c940`；F32 manifest/payload
SHA256=`1ad6d9edfb1037564c3aec533904ed8d48bc4614a7bbcb221a56c9a244d802c7` /
`480c57bff42e28cdfdf5ba260978234ee98f3d1169407c2523beeba76d39e834`。

F32 artifact 的 scalar count=`684352`，raw/serialized payload bytes=`2737408/2739448`，candidate
family=`[16]`。selection/confirmation max prediction difference=`7.8160793304e-7/7.8570211703e-7`，
normalized L2=`1.5589117128e-7/1.5627624021e-7`，aggregate MSE absolute drift=
`8.2576945282e-11/2.8491427967e-11`，relative drift=`3.9186724995e-9/1.3386508868e-9`。
Audit `35/35` gates passed，包含 exact cast identity、strict numerical tolerances、RMAL Gate-vector
identity、fresh reload、CPU boundary、state/no-update、no-bank 与 runtime independence。最终为
`passed / frozen_mean_preserving_data_aware_rank16_float32_numerically_qualified`，
`integration_only=true`；唯一下一动作是 `manual_review_MPCDAR_R16_F32_NQ_result`，不授权
GPU/production/default switch、decoder、training、generation 或 promotion。Bridge
refresh/status=`invalid`，原因是既有目标目录的 `.chatgpt-sources-manifest.json` 不合法；未删除、
覆盖或上传该 Sources 包。

## H-LOMA-CBOGT-S9V-FULL-PATH-VIS（2026-09-02）

S9V 在 Windows-native 工作区完成冻结 RMAL confirmation panel 的 descriptive-only full-path
visualization。S9V closeout commit=`013b53a3b5f46cf37b955a224b2701e7f9bcac7f`；权威 manifest 为
`outputs/20260902_v2/visualizations/scaled_chen_full_path_reconstruction_v1_attempt_01/visualization_manifest.json`，
SHA256=`ce6eee49822ddee7c3c28c13a6e0e108d499478984c498f2f51b909bebead749`，status=`completed`、
classification=`descriptive_visualization_completed`、forward consistency max residual=`0`，26 files。
输出未加入 Git，model/OLS state unchanged；S9V 不产生性能、manifold、training、generation 或
promotion 权限。定向测试=`9 passed`，父级/全套回归在该 checkpoint 通过；Bridge 仍为
`invalid / cloud_sources_sync_blocked`，未上传。S9V 后下一动作为 S10 implementation contract。

## 迁移后当前工作区登记（2026-09-03）

本登记只更新迁移后 Windows 工作区的当前状态，不改写 2026-08-30 的迁移、归档或恢复
事实。当前分支为 `research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`，当前 HEAD
为 `daf9d7f5871eedc8950833e90ca2ff95d3ad56e6`；本次起始工作树 clean。新增的诊断性
MMF M6 operator 位于 `src/ldm_project/v2/optics/nested_multimode_fiber_operator.py`，
导出位于 `src/ldm_project/v2/optics/__init__.py`，专项测试位于
`tests/v2/test_nested_multimode_fiber_operator.py`；SMF/FM F/formal path、训练、CUDA、
outputs 和冻结 evidence 均未触碰。

当前验证为 M6 `9 passed`、SMF `4 passed`、FMF M2 `8 passed`，changed modules/tests 的
compileall、Ruff、`git diff --check` 通过。FMF 回归为 Windows ACL 下显式临时根的
bounded test invocation。MMF M6 仅作为
`synthetic_nested_mmf_m6_second_order_identity_transport`、`diagnostic_only` 的人工审查
输入；其拒绝决策消费者为后续 VIS1、训练和 formal promotion gate，均不得由本实现自动
授权。唯一下一动作为 `manual_review_synthetic_mmf_m6_implementation`，治理 flags
`mainline_state=closed_at_current_evidence_boundary`、`training_authorized=false`、
`formal_promotion=false`、`generation_readiness=not_ready` 未改变。

## H-LOMA-CBOGT-S10-MPCDAR-R16-RUNTIME-PQ（2026-09-03）

S10 在 S9V closeout 后以 starting HEAD=`8680c12be955df72906b955433e6daeabfdfec4a` 完成冻结
S9 MPCDAR rank-16 float32 artifact 的 CPU-only、readout-only runtime/performance characterization。
implementation/fix commits=`dd1c38fa965e32deedd655f00f3cd974e1534cd8` /
`488f12fbcf19e3663c7aa685bc4d4ad717db2a30` /
`d0655652cd86bdc9f3852a9bbf4faa45f59f069b`；S9/S9V parent、frozen runtime、model、OLS、probe、
detector 与既有 evidence 未修改。

权威 write-once attempt-02 为
`outputs/20260902_v2/runs/lightgen_scaled_chen_mpcdar_rank16_runtime_performance_qualification_v1_attempt_02/`；
validation/audit/compact SHA256=`8293e549b1ae48447e28be924406525aa03fa8843c0f1a71fa1ec21e3f5894c5` /
`7b519b3a75cdf59ecfe2678d46a8f74ae3084cb7b9de90c95763efecc0f2f4ee` /
`ded0959097c88a74031edc1c0aa925d1a95a2df7f1abc5c6d9ba6ca7abdc3173`；attempt-01 按 write-once 保留
但不消费。结果为 `passed / frozen_rank16_float32_runtime_performance_characterized`，audit=`26/26`。

固定 CPU fixture 为 dense finite `float32[32,256]`、seed=`20260902`、SHA256=
`45ca137458856a7305ecf9500220c120aa5b4947e6f58d476b791e49b6337ab6`。冻结公式为
`centered=z-mu_fit; h=centered@T16; prediction=c_mean+h@V16.T`；B=`1/8/32` direct formula 与
repeated-call correctness residual 均为 `0`。fresh-process initialization 的
manifest/payload/consumer/total timing=`0.2792/9.1288/20.8566/30.286 ms`，fresh-process wall=
`3657.2196 ms`，不作 true cold-disk claim。固定 50 warmup、200 measured repetitions、
`time.perf_counter_ns` 的 batch mean latency B=`1/8/32` 为 `0.5601775/0.8914525/3.277882 ms`；
per-sample mean=`0.5601775/0.1114315625/0.1024338125 ms`；mean throughput=
`1836.8588342/9338.7841702/9874.3015643 samples/s`。完整 quantile/min/max 在 compact evidence。

环境实测 Python=`3.11.16`、PyTorch=`2.5.1`、torch threads=`8/8`、logical cores=`16`、CPU model=
`AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD`；expected Windows=`10.0.26200.0`、observed=
`10.0.26100`，差异已记录；physical cores/RAM=`not_recorded`。storage manifest/serialized/raw-runtime
分别为 `3219/2739448/2737408` bytes；process memory/RSS=`not_recorded`。分析 MAC 为
factorized/full-W=`644096/10240000` per sample、ratio=`15.89825119236884`，不是 measured speedup；
GPU、full-W runtime、end-to-end pipeline=`not_evaluated`。

S10 定向/布局/相关回归=`30 passed`，Windows-native 全套 pytest=`1357 passed, 12 deselected,
25 subtests passed`，compileall、Ruff、diff-check 通过；唯一 warning 为既有 pytest cache ACL。
Governance flags 保持 reconstruction/generative false、generation_readiness=`not_ready`、
training/promotion=false。唯一下一动作是 `manual_review_MPCDAR_R16_RUNTIME_PQ_result`；不得训练、
decoder、generation 或 promotion。Bridge 仍为 `invalid / cloud_sources_sync_blocked`。

## FINAL_MAINLINE_CLOSEOUT：H-LOMA-CBOGT Scaled-Chen（2026-09-03）

本次记录把 RMAL → S7 MPCDAR → S8 F64 R16-IQ → S9 F32-NQ → S9V full-path visualization →
S10 runtime characterization 固定为当前主线的最终 authority chain。后续 consumer 只消费命名的
immutable parent；所有较早或失败 attempt 按 write-once 保留，不以 earlier attempt 替代 later
authority。本节是文档治理收口，不运行模型/CUDA、不生成 evidence、不修改 outputs 正式产物、
SQLite、历史 transcript 或训练/推理代码。

最终状态为：Current phase=`MAINLINE CLOSEOUT`；Scientific diagnosis=`closed`；Repair=`confirmed`；
Integration=`qualified`；F32=`qualified`；Visualization=`completed`；Runtime performance=`characterized`；
`mainline_state=closed_at_current_evidence_boundary`；current mainline next scientific task=`none`。
假设/实验 thread 为 `closed / resolved to current evidence boundary`，历史假设不删除，历史
`next action` 仅作 dated provenance。未来若明确授权，独立 track 为 `REAL-HARDWARE DEPLOYMENT`，
显式 task name 为 `REAL_CMOS_RASPBERRY_PI_DEPLOYMENT`。

冻结结论为：task-relevant information survives existing optical path through CMOS；重建失败主要
由 mapping/readout misalignment 支持；simple locality/raw Euclidean W-SVD 不足；存在 useful
mean/DC material；centered detector information 更适合 fit-derived data-aware basis；registered
MPCDAR rank 16 是其 preregistered powers-of-two family 内 first confirmed candidate，但不是 minimum、intrinsic
或 true rank 16。冻结公式为 `centered=z-mu_fit; h=centered@T16; prediction=c_mean+h@V16.T`；
S9 F32 artifact 为 `mu_fit[256]`、`c_mean[40000]`、`T16[256,16]`、`V16[40000,16]`，
scalar count=`684352`，raw/serialized=`2737408/2739448` bytes。

允许的架构表述是：`optical front end → CMOS intensity sensing → lightweight frozen electronic
rank-16 readout`，即 hybrid optical-electronic reconstruction；MPCDAR 不是 native CMOS optical
image。未建立的边界包括 native optical target-like CMOS reconstruction、real physical CMOS、
Raspberry Pi benchmark、sim-to-real detector alignment、end-to-end hardware latency、GPU、full-W
runtime、quantization、generative manifold 与 generation readiness。

flags 继续固定为 `reconstruction_manifold_supported=false`、`generative_manifold_supported=false`、
`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false` 与
`closed_task_relevant_bottleneck_not_supported`。未来 tracks 必须独立注册：real CMOS/Pi deployment、
sim-to-real alignment、仅在另有理由时的 optional optimization、重新授权的 native optical redesign、
以及独立 generative-manifold/generation evidence。Bridge 已在最终 clean HEAD 上按工具合同执行
`refresh`/`status`：`refresh` fail-closed，`status=invalid`，确切原因是
`invalid Sources manifest: C:\Users\24567\Documents\Codex\codex-original-chatgpt-bridge\current\.chatgpt-sources-manifest.json`。
记录 `bridge_recovery_blocked=true`；未删除/覆盖现有 package，未自动上传。

## H-LOMA-CBOGT-FIBER-VIS1（2026-09-03，Windows-native，diagnostic-only）

在分支 `research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`、starting HEAD
`4d2933cb5b49302dd2114d49b56c905ef9d10ca7` 的 clean worktree 上完成已批准的 frozen horizontal VIS1。
S9V/RMAL/checkpoint/panel/S7-S9 parent evidence 在运行前逐项复核；checkpoint SHA256=
`ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de`，RMAL config SHA256=
`c50b5b2d12830c40fbe619f137aca40f36f799f277e637d1ec20ac7e3403859f`。只比较 native SMF 100、existing FMF M2 200、
synthetic nested MMF-M6 600 mode coefficients；三臂共用同一 encoder/pre-OLS tensor，decoder、CMOS、frozen MPCDAR R16 F32、
geometry、panel 和 display policy 均固定。未训练、未 refit、未 rank/seed/fiber sweep、未运行 evaluator、未修改
S7/S8/S9/S9V/S10、MPCDAR 或 formal runtime。

权威 write-once run 为
`outputs/20260903_v2/runs/lightgen_scaled_chen_fiber_horizontal_visualization_v1_attempt_06/`；Figure A/B、
`fiber_contracts.json`、`comparison_manifest.json`、`comparison_metrics.json`、`panel.pt` 与图形 QA artifacts 均已生成。
attempt-01 至 attempt-05 保留但不消费。MMF-M6 extra families 仅标注为 synthetic mode families，未作 physical LP-mode 或
real-MMF claim；modal metadata 保留实际 flat ordering，panel 不导出完整 600-mode bank。

固定 CUDA 10-sample replay elapsed=`2.2810000000008586 s`、peak allocation=`1748934144` bytes；learned model state hash
before/after identical，arm operators parameter-free。最终 scoped targeted suite=`37 passed`；compileall、Ruff、diff-check
通过。alignment/collision/text QA 分别为两张 PASS/两张 PASS/两张无低于 5 pt 文本；source validator 为
`20 PASS, 0 FAIL, 1 intentional WARN`，WARN 仅说明固定单 panel 不生成统计 uncertainty encoding。Gate 为
`descriptive_fiber_visualization_completed`，不改变 mainline closure、generation/training/promotion flags；唯一下一动作为
`IMPLEMENT_H_LOMA_CBOGT_FIBER_VIS1`。Bridge 仍 `invalid / cloud_sources_sync_blocked`，没有执行 refresh/fix/upload。

## H-LOMA-CBOGT-FIBER-VIS2（2026-09-03，Windows-native，diagnostic-only）

VIS2 canonical clear 0–9 full-pipeline visualization 已完成。branch=
`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`，starting/ending HEAD=
`4d2933cb5b49302dd2114d49b56c905ef9d10ca7`；执行前已核验上一轮 VIS1 的预期 dirty paths，未发现额外用户修改。
执行后只保留两份状态文档、VIS1 renderer/test 与 VIS2 renderer/test 六个预期路径；未提交，HEAD 未改变。

Stage A 使用同一 MNIST train split、S9V confirmation interval `[55000,60000)`，候选按 dataset index ascending、
每 digit 32 个，选择只基于 raw input。人工复核通过的 canonical indices 为
`0:55297, 1:55310, 2:55040, 3:55054, 4:55053, 5:55322, 6:55066, 7:55052, 8:55165, 9:55120`；
clear panel manifest SHA-256=`dbf9aa6741d29e264543f387edf7a4125893ec282c7a3897a57682d2e90388b9`。

上一轮 authority 为 write-once attempt-08：
`outputs/20260903_v2/runs/lightgen_scaled_chen_fiber_clear_full_pipeline_visualization_v1_attempt_08/`。
attempt-01 至 attempt-07 保留但不消费；attempt-06 为排版修正后的 partial run，attempt-07 为尚有分片页标题
碰撞的 completed run，均未覆盖或替代任何既有 evidence。attempt-08 manifest 要求 15/15 artifact、58 output files、
`396579365` bytes；包含 clear contact/approval、Figure 0、full-pipeline index 与 0–4/5–9 split pages、fiber close-up、
CMOS/reconstruction、SMF/FMF/MMF-M6 full paths、comparison/visualization manifests 与 panel payload。

三臂固定为 SMF 100、FMF M2 200、synthetic MMF-M6 600 coefficients；frontend 只 replay 一次，三臂共享同一个
pre-OLS tensor，decoder/relay/CMOS/frozen MPCDAR R16 F32 不变。MMF-M6 extra families `ETA1/XI2/XIETA/ETA2` 为
synthetic-only labels，不支持 physical guided modes、real propagation、capacity 或 ranking claim。full replay 为
CUDA 10-sample，elapsed=`2.7030000000013388 s`、peak allocation=`3095443968` bytes、model state unchanged=`true`。

QA：VIS2/VIS1/S9V/SMF/FMF-M2/MMF-M6 targeted tests=`44 passed`，唯一 warning 是既有 pytest cache ACL；compileall、
Ruff、diff-check 通过。10/10 alignment PASS；10/10 PDF 无低于 5 pt 文本（minimum=`5.0 pt`）；collision audit 无 FAIL，
4 个 modal-label figures 的 66 个 edge WARN 已人工复核；source validator=`19 PASS/0 FAIL/2 intentional WARN`。
治理仍为 reconstruction/generative=false、generation_readiness=`not_ready`、training/promotion=false、
`mainline_state=closed_at_current_evidence_boundary`；VIS1 unchanged，Bridge 仍 `invalid / cloud_sources_sync_blocked`。
Gate=`canonical_clear_full_pipeline_visualization_completed`。该 attempt 随后由下方 panel-only compact attempt-09
替代为当前可审查输出；唯一下一动作：`manual_review_VIS2_figures`。

## H-LOMA-CBOGT-FIBER-VIS2 compact rerender（2026-09-03，Windows-native，diagnostic-only）

已批准的 VIS2 attempt-08 `panel.pt` 被直接复用，完成 compact arm-separated attempt-09；未重新 CUDA replay、未改变
optical data、shared frontend、共同 p99.5 scales、phase mask、三臂 contracts 或 VIS1 attempt-06。branch 与
starting/ending HEAD 均为 `research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1` /
`4d2933cb5b49302dd2114d49b56c905ef9d10ca7`，无额外用户修改，未提交 Git commit。

最终输出：
`outputs/20260903_v2/runs/lightgen_scaled_chen_fiber_clear_full_pipeline_visualization_v1_attempt_09/`；source panel
SHA-256=`1ff94b0c14ba166531c81c781f0305cd10ae8f6d7efcebe04d457777a9d3f8d3`，panel loader 已实际校验 panel schema、
task id、APPROVED review、clear-manifest/source manifest hashes、comparison/visualization bindings 与三臂 payload。
layout schema=`compact_arm_separated_v2`，主图 21 个 bundle，digits 0–4/5–9 分页，三阶段组和三种 arm 独立页面，
空间场 `aspect=equal`，modal family×port 为显式非空间矩阵，SMF/FMF/MMF-M6 使用蓝/绿/橙色边界，MMF-M6 synthetic
family labels 保持显式。输出 115 files、70354448 bytes，PNG/PDF/SVG=`22/22/44`，alignment JSON=`22`，无 TIFF、
隐藏文件或 `.tmp` 残留。

QA：VIS2 panel/布局回归 `13 passed`；22/22 alignment PASS；22/22 PDF collision PASS（0 WARN/0 FAIL）；22/22 PDF
text audit 无低于 5.0 pt 文本（minimum=5.0 pt）；Nature source validator=`17 PASS/0 FAIL/4 intentional WARN`，其中
PNG-only raster、300 dpi floor、Stage-A contact-sheet 宽度和固定样本 uncertainty WARN 均是明确的诊断/输出约束；
compileall、Ruff、`git diff --check` 均通过。

按批准范围精确删除 VIS2 attempt-01–attempt-08 与 VIS1 attempt-01–attempt-05，释放 `2942680973` bytes（约
`2.7406 GiB`）。当前仅保留 VIS1 attempt-06 与 VIS2 attempt-09，合计 `297510614` bytes；无隐藏文件或 `.tmp` 残留。
治理仍为 `reconstruction/generative=false`、`generation_readiness=not_ready`、`training/promotion=false`、
`mainline_state=closed_at_current_evidence_boundary`、Bridge=`invalid / cloud_sources_sync_blocked`。Gate=
`canonical_clear_full_pipeline_visualization_completed`；唯一下一动作：`manual_review_VIS2_figures`。

## H-LOMA-CBOGT-FIBER-VIS2 attempt-10（2026-09-03，Windows-native，diagnostic-only）

attempt-09 的已捕获 `panel.pt` 被直接复用，完成 attempt-10 分层 panel-only rerender；未重新 CUDA replay、未改变
optical tensor、display scale、shared frontend、phase mask、三臂 contracts 或 VIS1 attempt-06。attempt-09 保留为
不可变审计源，attempt-10 为当前结果；未提交 Git commit。branch 与 starting/ending HEAD 为
`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1` / `4d2933cb5b49302dd2114d49b56c905ef9d10ca7`，
无额外用户修改。

输出目录为
`outputs/20260903_v2/runs/lightgen_scaled_chen_fiber_clear_full_pipeline_visualization_v1_attempt_10/`，目录分为
`digit_visualization`、`pipeline_overview`、`fiber_information/{SMF,FMF,MMF-M6}` 和 `metadata_qa`。共 `115` 个文件、
`70212115` bytes；主图 21 bundles，PNG/PDF/SVG=`22/22/44`，alignment JSON=`22`，无 TIFF、隐藏文件或 `.tmp`。
source/target panel SHA-256 均为 `1ff94b0c14ba166531c81c781f0305cd10ae8f6d7efcebe04d457777a9d3f8d3`；
layout schema=`organized_compact_arm_separated_v3`，manifest 显式绑定 source manifests 并标记
`supersedes=attempt_09`。所有 row labels 为水平 rotation=`0`；image-space、optical-space 与 modal matrix 的
origin/rotation 策略均显式记录，空间场保持 `aspect=equal`。

最终 QA：VIS2 回归=`19 passed`；VIS2/VIS1/S9V/SMF/FMF-M2/MMF-M6 targeted suite=`64 passed`（唯一 warning 为既有
pytest cache ACL）；22/22 alignment PASS；22/22 PDF collision PASS（0 WARN/0 FAIL）；22/22 PDF text audit 无低于
5.0 pt 文本（minimum=`5.0 pt`）；Nature source validator=`17 PASS/0 FAIL/4 intentional WARN`。代表性的
pipeline overview、SMF frontend、FMF fiber decoder、MMF-M6 fiber decoder/detector 页面已实际查看并通过版式检查。

治理保持 `reconstruction/generative=false`、`generation_readiness=not_ready`、`training/promotion=false`、
`mainline_state=closed_at_current_evidence_boundary`、Bridge=`invalid / cloud_sources_sync_blocked`；本轮未执行
CUDA、训练、refit、rank search、formal runtime 或 bridge。Gate=`canonical_clear_full_pipeline_visualization_completed`；
唯一下一动作：`manual_review_VIS2_figures`。

## H-LOMA-GEN-G1-SA prior support accessibility observer（2026-09-03）

完成一次 Windows-native、diagnostic-only、no-update 的 G1-SA observer；未重新 fiber comparison，未进入 decoder/CMOS/
MPCDAR，未训练、refit、rank search、formal runtime 或 bridge。起始与结束 HEAD 均为
`d07fdaa17aa91d9f3e5498e9b1f10d91db766777`，branch=`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`；
正式 replay 前 worktree clean，当前新增路径仅为 G1 config、observer 和 targeted test。

按 attempt-05 MRA lineage 固定 step-5000 checkpoint/calibration，使用 exact 100D `z_j=abs(c_j)^2/s_j`。完整 train
support=`55000`（`[0,55000)`），horizon heldout query=`5000`（`[55000,60000)`），unconditional truncated-normal prior
query=`5000`，seed=`1237`，距离为 exact chunked Euclidean / `sqrt(100)`。当前可审查输出为
`outputs/20260903_v2/runs/lightgen_scaled_chen_g1_prior_support_accessibility_v1_attempt_04/`；attempt-01 保存原始
CUDA measurement，attempt-04 仅从同一 distance artifact 无 CUDA 重绘 figures，write-once 保留中间 attempts。共
`12` files、`294949` bytes；manifest SHA-256=`257e6b786714b6f0a76fe93e4bba6ddf2f6bc3b691d0ef3f582ebd7c7521f239`，
distance SHA-256=`05002920d1b73ab8dd1243ce93f44659ba95ad5d4eb35e3f4e737c0f9d8b2c1b`，无 TIFF/隐藏文件/`.tmp`。

结果：`H50=0.0006687190`、`H95=0.0011482780`；`P50=1.1241731`、`P95=1.2017229`；`R50=1681.0843`，95% CI
`[1657.9121,1705.0805]`；`R95=1046.5435`，95% CI=`[1019.9279,1063.9357]`；prior exceedance of H95=`1.0`
（descriptive-only）。Gate=`P4_SUPPORTED`，含义是 registered prior 与真实 encoded support 明显不对齐，不是
decoder/generation/training 成功。冻结审计 state digest before/after 相同，optimizer/backward/refit/checkpoint/
evaluator=`false/false/false/false/0`；runtime=`372.0281 s`，peak CUDA=`1574747648` bytes。

QA：G1 targeted=`8 passed`；compileall、Ruff、diff-check 通过；PDF text audit=`2/2`（均无低于 5 pt，minimum=`5.6 pt`），
collision=`2/2 PASS`（0 WARN/0 FAIL），两张 PNG 已实际检查。治理不变：
`mainline_state=closed_at_current_evidence_boundary`、`generative_manifold_supported=false`、
`generation_readiness=not_ready`、`training_authorized=false`、`formal_promotion=false`、
`bridge=invalid / cloud_sources_sync_blocked`。唯一下一动作：`PLAN_GENERATIVE_TRAINING_FOR_SUPPORT_ALIGNMENT`，仅为
规划动作，不启动训练。

## H-LOMA-CBOGT-FIBER-VIS2 three-photo matched-input comparison attempt-11（2026-09-03，Windows-native，diagnostic-only）

为解决 attempt-10 的可读性问题，复用未修改的 attempt-10 `metadata_qa/panel.pt` 完成 panel-only 三页重绘；未重新 CUDA
replay、未改变 optical tensor、display scale、shared frontend、phase mask、三臂 contracts 或 VIS1 attempt-06。
attempt-10 保留为 source，attempt-11 使用独立目录。branch=`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`，
starting/ending HEAD=`d07fdaa17aa91d9f3e5498e9b1f10d91db766777`；G1 prior-support 未提交路径与既有状态文档作为受保护基线，
未发现额外 dirty path。

输出为
`outputs/20260903_v2/runs/lightgen_scaled_chen_fiber_clear_full_pipeline_visualization_v1_attempt_11/`，layout schema=
`same_digit_three_arm_comparison_v1`。三张主图按标准数字分为 `0–3`、`4–6`、`7–9`；每页四列固定为
`Standard digit input | SMF output | FMF output | MMF-M6 output`，每行三种光纤复用同一个输入 position。输入字段为
`images_28x28`，唯一输出字段为 `mpcdar_rank16_float32`，图中仅显示 digit label，index 写入 manifest。SMF/FMF/MMF-M6
边框为蓝/绿/橙，空间轴等比例显示；MMF-M6 synthetic family 解释边界不变。

attempt-11 共 `25` files、`38798939` bytes，无 TIFF、隐藏文件或 `.tmp`。source panel SHA-256=
`1ff94b0c14ba166531c81c781f0305cd10ae8f6d7efcebe04d457777a9d3f8d3`，manifest 记录 source visualization/comparison
bindings；comparison manifest SHA-256=`e9b3c329bd3b30449cccf618d54d3b9b22c3edd04e46adb2057597c0ca4a3a99`，
visualization manifest SHA-256=`2b457baa83b85620c9ea9374bf5b6bc47389fadcf94991c95064ab325ebc07b`；并记录
`render_only_from_panel=true`、`cuda_replay=false`、`optical_data_changed=false`、`supersedes=attempt_10`。
三页 PNG/PDF/SVG 与各自 alignment JSON/SVG 为唯一主图输出，approved clear-panel artifacts 和 panel 保留在
`metadata_qa/` 供审计。

QA：VIS2 renderer targeted=`23 passed`；Windows-native compileall、Ruff、diff-check 通过；alignment=`3/3 PASS`，PDF text
audit=`3/3`（minimum=`5.4 pt`），collision=`3/3 PASS`（0 WARN/0 FAIL）。alignment canvas=`504 pt`，PDF tight-bbox 宽度
`413.4/430.5 pt` 或 `430.5 pt`，PNG 尺寸为 `1722×1399`、`1793×1399`、`1793×1399`；三页均已实际查看。Nature source
validator 非 strict=`18 PASS/0 FAIL/3 WARN`，warning 为既有 renderer 的诊断性静态启发式，不对应本次三页输出。

治理保持 `mainline_state=closed_at_current_evidence_boundary`、`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false`、`generation_readiness=not_ready`、`training_authorized=false`、
`formal_promotion=false`、`bridge=invalid / cloud_sources_sync_blocked`；未运行 CUDA、训练、refit、rank search、formal
runtime 或 bridge。Gate=`canonical_clear_full_pipeline_visualization_completed`；唯一下一动作固定为
`manual_review_VIS2_figures`。

## H-LOMA-CBOGT-FIBER-VIS2 reference-style process-flow rerender attempt-14（2026-09-03，Windows-native，diagnostic-only）

为恢复参考图式的完整流程表达，复用未修改的 attempt-10 `metadata_qa/panel.pt`，新增并执行
`--render-process-flow-from-panel` panel-only 入口。未重新 CUDA replay，未改变 optical tensor、display scale、
shared frontend、phase mask、三臂 contract 或 VIS1 attempt-06 authority；attempt-10/11 保持不可覆盖，attempt-12/13
作为未通过严格 collision QA 的 write-once 布局迭代保留。当前候选输出为
`outputs/20260903_v2/runs/lightgen_scaled_chen_fiber_clear_full_pipeline_visualization_v1_attempt_14/`。

layout schema=`same_digit_three_arm_process_flow_v1`，三页固定为标准数字 `0–3`、`4–6`、`7–9`。每页按参考图
组织完整流程列：`Input → SLM command → SLM output → Encoder in → Encoder L1 → Pre-OLS → Modal →
Post-fiber → Decoder L1 → Decoder L2 → CMOS field → CMOS intensity → MPCDAR R16`；每个数字在 SMF、FMF、
MMF-M6 三条上下对齐的彩色 lane 中各有相邻 amplitude/phase 行。同一 panel position 复用同一个标准输入和
dataset index；shared frontend 在三条 lane 中重复显示以便直接横向对照。SMF/FMF/MMF-M6 使用蓝/绿/橙边框与
行标签区分，Modal 是显式 family×port 非空间矩阵，MMF-M6 的 `ETA1* / XI2* / XIETA* / ETA2*` 保持 synthetic
标识；不作 fiber ranking、physical MMF、capacity、reconstruction-manifold、generation、training 或 promotion
结论。

attempt-14 共 `25` 个文件、`56214710` bytes：3 个 PNG、3 个 PDF、3 个主 SVG、3 个 alignment JSON、3 个
alignment SVG，以及 `metadata_qa/` 下 9 个 approved source/manifest artifacts。无 TIFF、隐藏文件或 `.tmp`；
PNG 尺寸为 `3363×4204`、`3366×3303`、`3366×3303`，PDF 均为单页。source panel SHA-256 保持批准值
`1ff94b0c14ba166531c81c781f0305cd10ae8f6d7efcebe04d457777a9d3f8d3`；source comparison/visualization manifest
SHA-256 分别为 `d4229e4f966f911fee510ed4bcfa385eb46821ecf84289c0e6b7a745f679fdd9` 与
`8dcc62732b213c42cd12539b6f9329310ae2ec252113261154ba5f9fe2ed8648`。attempt-14 comparison/visualization
manifest SHA-256 分别为 `71ed3ccd75e7ee2748fd30ba6bc4f60d9fabfe7f1bcd6ec71d22619144e08f1c` 与
`705ef3d7e041219dac3090ede56701d5f41767860a985ef87c5e5dfb481ce1f5`；manifest 标记
`render_only_from_panel=true`、`cuda_replay=false`、`optical_data_changed=false`、`supersedes=attempt_11`。

QA：Windows-native relevant suite=`51 passed`；compileall、Ruff、`git diff --check` 通过。3/3 alignment audit
为 PASS（0 fail/0 warn），3/3 PDF text audit 无低于 `5.0 pt` 文本，3/3 strict collision audit 为 PASS（0 fail/
0 warn）；三张 PNG 已实际查看。Nature source validator=`18 PASS/0 FAIL/3 WARN`，WARN 仅为既有 renderer 的
300 dpi、宽度启发式和固定诊断样本 uncertainty 启发式，不改变本次实际输出的无 TIFF 与严格图形 QA 结论。

执行前后 branch 均为 `research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`，starting/ending HEAD 均为
`d07fdaa17aa91d9f3e5498e9b1f10d91db766777`。worktree 仍只有受保护 G1 prior-support config/observer/test、两份
状态文档、当前 VIS2 renderer/test；未发现额外用户修改，未创建 Git commit。治理保持
`mainline_state=closed_at_current_evidence_boundary`、`reconstruction_manifold_supported=false`、
`generative_manifold_supported=false`、`generation_readiness=not_ready`、`training_authorized=false`、
`formal_promotion=false`、`bridge=invalid / cloud_sources_sync_blocked`；未运行 CUDA、训练、refit、rank search、
formal runtime 或 bridge。唯一下一动作固定为 `manual_review_VIS2_figures`。

## H-LOMA-CBOGT-FIBER-VIS2 reference-style process-flow rerender attempt-15（2026-09-03，Windows-native，diagnostic-only）

按人工反馈将 attempt-14 的按光纤整组排列改为同一数字内交错排列。复用未修改的 attempt-10
`metadata_qa/panel.pt`，通过 `--render-process-flow-from-panel` panel-only 入口生成
`outputs/20260903_v2/runs/lightgen_scaled_chen_fiber_clear_full_pipeline_visualization_v1_attempt_15/`；
attempt-10 与 attempt-14 保持不可覆盖，attempt-15 supersedes attempt-14。未重新 CUDA replay，未改变
optical tensor、display scale、shared frontend、phase mask、三臂 contract 或 VIS1 attempt-06 authority。

layout schema=`same_digit_interleaved_three_arm_process_flow_v1`，artifact layout schema=
`vis2_process_flow_interleaved_v1`，row order=`digit_then_arm_then_amplitude_phase`。三页固定为数字
`0–3`、`4–6`、`7–9`；每个数字严格显示 `SMF amplitude/phase → FMF amplitude/phase → MMF-M6 amplitude/phase`，
再进入下一个数字。完整参考流程列为 `Input → SLM command → SLM output → Encoder in → Encoder L1 → Pre-OLS →
Modal → Post-fiber → Decoder L1 → Decoder L2 → CMOS field → CMOS intensity → MPCDAR R16`。SMF/FMF/MMF-M6 使用
蓝/绿/橙边框和行标签；空间图等比例，Modal 保持显式 family×port 非空间矩阵，MMF-M6 synthetic family 标签
显式保留；不作 fiber ranking、physical MMF、capacity、reconstruction-manifold、generation、training 或
promotion 结论。

attempt-15 共 `25` 个文件、`56234887` bytes，PNG 尺寸为 `3363×4204`、`3366×3303`、`3366×3303`，PDF 均为
单页；无 TIFF、隐藏文件或 `.tmp`。source panel SHA-256=`1ff94b0c14ba166531c81c781f0305cd10ae8f6d7efcebe04d457777a9d3f8d3`；
source comparison/visualization manifest SHA-256 分别为 `d4229e4f966f911fee510ed4bcfa385eb46821ecf84289c0e6b7a745f679fdd9`
与 `8dcc62732b213c42cd12539b6f9329310ae2ec252113261154ba5f9fe2ed8648`；attempt-15 comparison/visualization
manifest SHA-256 分别为 `2d9946b29f1bd24a34081e0cef3dde6a872f894c3d5d2681ac9752198148196a` 与
`2cfe75f7bf6248504e65ecbf0a769508eba968a38408eb695a23ba95970509f3`。manifest 记录
`render_only_from_panel=true`、`cuda_replay=false`、`optical_data_changed=false`、`rows_per_digit=6`、
`row_records` 与 `supersedes=attempt_14`。

QA：交错行序回归测试通过，Windows-native relevant suite=`52 passed`；compileall、Ruff、`git diff --check` 通过；
3/3 alignment=`PASS`（0 fail/0 warn），3/3 PDF text audit 最小字号=`5.0 pt`且无低于阈值文本，3/3 strict
collision=`PASS`（0 fail/0 warn），三张 PNG 已实际查看。Nature source validator=`18 PASS/0 FAIL/3 WARN`，WARN
为既有 renderer 的静态启发式，不改变本次严格图形 QA 结论。

执行前后 branch 均为 `research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`，starting/ending HEAD 均为
`d07fdaa17aa91d9f3e5498e9b1f10d91db766777`；worktree 只有受保护 G1 prior-support config/observer/test、两份
状态文档、当前 VIS2 renderer/test，未发现额外用户修改，未创建 Git commit。治理保持 diagnostic-only、training
unauthorized、formal promotion false、bridge invalid；本轮未运行 CUDA、训练、refit、rank search、formal runtime
或 bridge。唯一下一动作固定为 `manual_review_VIS2_figures`。

## H-LOMA-CBOGT-S9V-FULL-PATH-VIS upright eight-stage rerender attempt-02（2026-09-03，Windows-native，diagnostic-only）

保留不可覆盖的 full-path visualization attempt-01，基于相同冻结 step-5000 checkpoint、RMAL/S7/S8/S9 parent
chain 和 confirmation panel 执行一次受控 CUDA replay；未训练、未 refit、未 rank search、未修改光学参数或
checkpoint。新输出为 `outputs/20260902_v2/visualizations/scaled_chen_full_path_reconstruction_v1_attempt_02/`；
attempt-01 manifest SHA-256=`ce6eee49822ddee7c3c28c13a6e0e108d499478984c498f2f51b909bebead749`，新 manifest
SHA-256=`393ebefe6c11537f5e1841a905eb0eed4365d7db57879cd823d477338b146703`。

新的 `full_optical_path_eight_stage_upright_v1` 主图固定 8 个阶段：`SLM output`、`Front 4f`、`Encoder layer 1`、
`Encoder layer 2 / Pre-OLS`、`OLS output / Decoder input`、`Decoder layer 1`、`Decoder layer 2`、
`Camera incident field`。每个数字显示 amplitude/phase 两行，页面为 `0–4` 和 `5–9`；主图宽度约 `7.18 in`，
PNG 尺寸均为 `2153×2387`。Supporting reconstruction/repair/error/F64-F32 figures 仅保持原有语义。attempt-02
共 `26` 个文件、`13413276` bytes，仅含 PNG/PDF/SVG/JSON，无 TIFF、隐藏文件或 `.tmp`。

方向合同明确：IdealAmplitudeSLM 不改变 command 的空间坐标，旧图 SLM output 倒置是 renderer 的 display-only
180° rotation；真实前向/后向 4f relay 的倍率分别为 `-0.4` 与 `-2.5`，两次倒像的净倍率为 `+1.0`。新图只
摆正绘图副本，不改变 raw optical tensor、CMOS intensity、reconstruction 或实际生成内容，并在 manifest 中
区分 display rotation 与 native coordinate。

QA：相关 v2 optical/full-path tests=`32 passed`；compileall、Ruff、`git diff --check` 通过；5/5 alignment、
5/5 PDF text（最小字号 `5.0 pt`）、5/5 strict collision 均 PASS（0 fail/0 warn）；两张主图 PNG 已实际查看。
Nature source validator=`20 PASS/0 FAIL/1 WARN`，WARN 仅为既有 legacy export/宽度启发式。attempt-01 26 个文件
逐文件 SHA-256 与运行前一致；starting/ending HEAD=`d07fdaa17aa91d9f3e5498e9b1f10d91db766777`，branch=
`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`。当前 dirty path 仅为既有 G1 prior-support 与本次
renderer/test/config/状态文档，未发现额外用户修改，未创建 Git commit。治理保持 diagnostic-only、training
unauthorized、formal promotion false、bridge invalid；唯一下一动作固定为 `manual_review_FULL_PATH_VIS_result`。

## H-LOMA-GEN-P1-C6-RSC（2026-09-03）

Windows-native environment `D:\ANACONDA\envs\optical-generative-models-win` 在 branch
`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`、HEAD
`2fcd16a0bef7de2a6483f5ec42070a895bbcbcac` 上完成一次受控 CUDA raw-scale canary。校验闭合旧
calibration v1 SHA、历史 dataset manifest 内容 hash、calibration indices、mode bank、horizon/formal
config、review、preflight 与 validation；仅允许 Windows dataset root 与 manifest self-hash 的已知路径差异。

新增诊断 helper 只暴露原 producer 的 pre-floor raw port powers，默认 `calibrate_model` 的 floor/clamp
行为保持不变。新增测试先经 baseline failure，再以 `42 passed` 通过；compileall、Ruff 与 `git diff --check`
通过。输出目录为
`outputs/20260903_v2/runs/lightgen_scaled_chen_p1_calibration_raw_scale_canary_v1_attempt_01/`，大小
`547210` bytes，输出树严格仅含 JSON/PT/PNG；无 TIFF、隐藏文件或 `.tmp`。artifact hash binding、state
digest equality 与 no-training flags 已独立核验，raw summary PNG 已实际查看。

结论为 `C6_CANARY_PASS`：当前 raw power 均为可表示的正 float32 数，absolute eps 高于真实 raw scale，
旧 eps-dominated artifact 可由 raw replay 精确解释；zero-phase 为注册 reference coordinate。治理继续为
`mainline_state=closed_at_current_evidence_boundary`、reconstruction/generative manifold unsupported、
generation not ready、training unauthorized、formal promotion false。下一步仅登记为
`IMPLEMENT_R2_UNIT_AWARE_CALIBRATION_REPAIR`，本记录不执行 repair。

## H-LOMA-GEN-P1-R2-UACR unit-aware calibration contract repair（2026-09-04）

在 branch=`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`、starting HEAD=`877f386657c7b408fdfd1964510a473782607d80`
上完成 calibration-only R2。消费不可变 C6 attempt-01 的 raw payload（manifest SHA-256
`dfe880fecf297099c8c573579c7c064887ac1ded9fbd7b745e5bbadefcafedfe`、raw payload SHA-256
`4a2ffbf26418d5540db357b20daa56e552e6d8202fbb24af8a8ad869b10ced69`）和 v1 calibration（SHA-256
`a9a871405e216597b2c6058e8c52cd86a242659f8f6729fe06b85d0ef0f32959`）。未重新 CUDA、未运行 optical
forward、未修改 C6/G1/P4/VIS evidence、未进入 replay 或训练路径。

R2 v2 contract=`scaled_chen_bogt_calibration.v2` / `unit_aware_raw_mean_v2`：每个 finite positive
raw `mean_cal(|c_j|^2)` 直接作为 registered `port_scale`，无 absolute eps/tiny/median/fallback
floor；raw 与 registered 字段严格一对一保存。显式 loader 绑定 v2 artifact SHA、dataset manifest、
calibration indices、mode bank 与 parent v1 SHA，并拒绝 v1 fallback；原 v1 reader 保持可读。

当前 write-once authority 为
`outputs/20260904_v2/runs/lightgen_scaled_chen_bogt_calibration_contract_repair_v2_attempt_02/`
（attempt-01 保留），包含 6 个 JSON/PT/PNG 文件、`159393` bytes，无 TIFF、隐藏文件或 `.tmp`。
Gate=`R2_CALIBRATION_REPAIR_PASS`；raw FP64 min/median/mean/max=`1.508997e-12 / 1.127986e-10 /
1.741773e-10 / 6.304158e-10`，registered/raw exact equal，z global mean=`0.9999999986`，
最大 per-port mean deviation=`1.451667e-7`。R2 定向及直接依赖回归=`48 passed`，compileall、Ruff、
diff-check 通过。治理保持 closed evidence boundary、reconstruction/generative unsupported、
generation not ready、training unauthorized、formal promotion false；唯一下一动作固定为
`PLAN_REPAIRED_COORDINATE_REPLAY`。

## H-LOMA-GEN-RCR-A repaired-coordinate MRA replay（2026-09-04）

初始 branch=`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`、HEAD=`ef1b336984b701f1905a06fc868c68c4b4892057`、
worktree clean。RCR-A 独立 harness/config/test commit=`c0bbe12db909d4ceb5176f72dbdb69df5c6c8ff5` 后，按
Windows-native CUDA 资源合同加载固定 step-5000 checkpoint 与 parent MRA attempt-05 panel；不重新选择样本，
不进入 decoder/relay/CMOS/MPCDAR/evaluator，不运行 Stage B G1-SA。

Stage A 的唯一 intervention 是 calibration.v1 → calibration.v2：new payload SHA-256=
`ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07`、R2 manifest SHA-256=
`0cf707fe4f146c1a1c3190256fad8ec8e4bc72e33674d374e75f40b161b3fce6`。selection repaired z mean/variance=
`1.0060573/1.5498618`，confirmation=`0.9250944/1.4546335`；Gaussian KL=`3.2833059/2.8498273`，
numerical truncated-normal KL=`3.2492599/2.7428405`；complex rank=`12.2361650/12.8739576` 保持旧
formal values，power rank=`7.2165284/7.3683505` 仅为 coordinate-dependent 描述。两组 mean 均在 `±0.10`
内，但 variance 为 prior `0.25` 的 `6.1994×/5.8185×`，variance-shape term 在两组均占 Gaussian-KL
分解主导；exact dominant assumption=`unconditional_truncated_normal_variance=0.25_with_lower=0.0`。

完整 write-once authority 为
`outputs/20260904_v2/runs/lightgen_scaled_chen_repaired_coordinate_mra_v1_attempt_02/`，manifest SHA-256=
`10a5469468357c8490446ff4c9d1fc86bc1041bba4cfa6fa51af4fc8f01f70da`，4 个 JSON 共 `53575` bytes；无 TIFF、
隐藏文件或 `.tmp`。attempt-01 因发布顺序错误保留 3 个中间 JSON，不作为 authority。Gate=
`MRA_PRIOR_CONTRACT_REVIEW_REQUIRED`，next action=`PLAN_PRIOR_SHAPE_CONTRACT_REVIEW`。RCR-A/直接依赖回归=
`46 passed`，compileall、Ruff、`git diff --check` 通过；state digest before/after 均为
`cdbf06f03b611f4df38324c1546686bc5fc614644a64130c95cd3a76012e27b5`，peak CUDA=`1599782400` bytes，runtime=
`39.8108 s`。治理仍为 closed evidence boundary、reconstruction/generative unsupported、generation not ready、
training/promotion=false。

## H-LOMA-GEN-T1-RCWV repaired-coordinate BOGT weight validity canary attempt-01（2026-09-04）

完成初始 clean preflight 后，在固定 step-5000 parent、注册三组 panel 与已批准 R2 v2 calibration
上执行 no-update CUDA gradient measurement；没有 optimizer、optimizer.step、training、checkpoint、
prior generation、decoder/CMOS/MPCDAR 或 refit。branch=`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`，
starting/ending HEAD=`47b9aa0896cae88516b3872e6f3355ff12434054`。

旧 alpha=`0.002996375650268832` 的 provenance、parameter subset、norm/ratio formula、batch/microbatch
与三组 panel identity 均闭合。注册旧 preflight effective ratios=`0.1102407/0.0855945/0.1081556`；同一
step-5000 parent 的 v1 replay=`0.1864269/0.1105396/0.1395263`；repaired v2 replay=
`1.9516949/1.5488551/1.5754461`，median/min/max=`1.5754461/1.5488551/1.9516949`。Gate=
`RC_ALPHA_TOO_STRONG`，仅说明该 registered alpha 在 repaired coordinate 下超出 `[0.05,0.40]`，不授权
训练或 generation，也不实施 weight repair。

write-once authority 为
`outputs/20260904_v2/runs/lightgen_scaled_chen_repaired_coordinate_bogt_weight_validity_v1_attempt_01/`，
4 个 JSON、`38814` bytes；artifact reference hashes、hidden/temporary-file 检查通过。optimizer recovery
state=`AVAILABLE_AND_BOUND`（只读），checkpoint/state digest 前后一致，所有 no-update isolation flags
通过。T1-RCWV targeted tests=`8 passed`；compileall、Ruff、`git diff --check` 通过。治理保持
closed evidence boundary、reconstruction/generative unsupported、generation not ready、training/promotion
false。唯一下一动作固定为 `PLAN_REPAIRED_COORDINATE_ALIGNMENT_WEIGHT_REGISTRATION`。
## 2026-09-04：H-LOMA-GEN-T1-RCAWR Windows-native closeout

在 branch=`research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1`、starting HEAD
`7d7bae60d78ebdf80615c08e19a0ab435bfd34d3` 的 clean preflight 后，新增并验证 RCAWR
registration script/config/focused tests 及本次状态记录。实现仅消费冻结 RCWV/R2/历史 BOGT
artifact，使用 Decimal analytic replay 登记
`alpha_repaired=0.00017905883237862847`；没有 CUDA、模型 forward、gradient measurement、
optimizer、training、checkpoint、prior generation、refit、decoder、CMOS 或 MPCDAR。

RCAWR Gate=`RCAWR_REGISTRATION_PASS`；predicted ratios=`0.1166303093/0.0925572168/0.0941462530`，
均在 `[0.05,0.40]`。attempt-01 因 manifest 发布顺序检查过早而只留下 4 个中间 JSON，保留但不消费；
正式 write-once output 为
`outputs/20260904_v2/runs/lightgen_scaled_chen_repaired_coordinate_bogt_weight_registration_v1_attempt_02/`，
只含 5 个 JSON，manifest 记录 source hashes、Set A/B provenance、state isolation 与 governance，
不覆盖任何既有 attempt。Windows-native RCAWR focused tests、compileall、Ruff 与 `git diff --check`
通过；最终 Git commit 与 clean-head 状态由本轮收尾核验返回。唯一下一动作是
`IMPLEMENT_GENERATIVE_ALIGNMENT_CANARY`，training/promotion 仍未授权。
