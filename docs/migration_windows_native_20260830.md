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
