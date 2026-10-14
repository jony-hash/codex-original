# AutoDL 操作流程

本页是 AutoDL 项目的正式操作说明。它固定连接后的工程步骤，不替每项研究决定模型起点、数据人口、训练预算、指标或科学阈值。

## 文件与运行位置

- Windows 本地源码权威：`D:\codex-original`。
- AutoDL 执行副本：`autodl-monthly:/root/autodl-tmp/codex-original`。
- 云端 Python：`/root/miniconda3/bin/python`。
- 2026-10-14用户固定后续训练与长数值验证在该AutoDL执行；本地负责代码准备、审阅和精确备份。科学更新仍按每轮授权预算和预检执行。
- 用 VS Code Remote-SSH 打开云端项目时，编辑器和终端读写的都是 Linux 云端副本。保存文件不会自动写回 Windows。
- 源码和配置用 Git 管理；数据、检查点、报告和样本读出用 SCP 传输。不要对同一文件同时在本地和云端修改，也不要开启自动双向同步。
- 只有当已批准的具体任务明确要求临时源码覆盖时，才可在云端保留原文件备份后逐个部署指定文件；部署前核对目标路径，部署后核对文件内容、云端 `py_compile` 和从仓库根目录执行的 `python -m ... --help`。这种覆盖只服务该任务，不同步整个脏工作树，也不替代常规 Git 源码管理；保留云端其他改动，不提交或推送。
- 云端源码改动先审阅具体文件，再合并回本地。训练结果保留在云端数据盘；关键结果按每项实验清单回传到本地的新备份目录。云端实例释放前先确认备份可读。
- SSH凭据只放在本地 SSH 配置和 `ssh-agent` 中。脚本用 `BatchMode`，不会提示或保存密码；先在 Windows PowerShell 用 `ssh-add -l` 确认密钥已由 agent 提供。

VS Code Remote-SSH 可以用于云端编辑、终端操作、日志与图片查看。当前 Codex 远程扩展的地区登录限制不影响 SSH、训练或本地 Codex 对报告的分析；它也不代表训练进程会被持续监控。Codex 做分析时要给出明确的日志、报告、图像或检查点路径。

## 每项任务开始前

先单独确定并写明：

1. 科学问题及结果将怎样改变下一步选择。
2. 明确的模型起点、配置、检查点、数据人口和各数据角色。
3. 同一生产入口的预检计划，包括真实更新、可训练/冻结组、评价器隔离、方向相关负控和保存后恢复核验。
4. 固定更新数、学习率、调度、批量、精度，以及最后测量的人口、指标、阈值和停止点。
5. 该轮独立结果目录与需回传文件清单。

这些都是每项实验的科学设计，操作脚本不替用户补默认值。A4700仍是研究基线，B5400仍是研究候选。GPU可用、空闲显存、包月时长和操作链验收都不授予训练或采用权限。

## 环境检查

### Codex 必须使用沙箱外网络通道

2026-10-14实测：同一条 `autodl_run.ps1 -Action Check` 在Codex受限沙箱内报SSH 255/主机名解析失败，在沙箱外退出0，Python、CUDA和项目导入均正常。先前据沙箱错误判断本机DNS/代理异常并要求用户修复连接的结论撤回；这些失败只证明当时的沙箱网络通道不可用。

Codex调用SSH/SCP、`autodl_run.ps1`的Check/Start/Status及`autodl_pull.ps1`时，显式设置 `exec_command.sandbox_permissions="require_escalated"`，沿用用户对该云端任务的授权。普通PowerShell终端直接使用下面命令即可，不需要修改Windows DNS、代理或SSH配置。网络失败应先区分沙箱、客户端解析、TCP、SSH认证和远端命令；沙箱错误不得归因为云服务器DNS，也不得触发未授权的本地替代运行。外部执行若被自动审批拒绝，单独报告审批原因。

该修复不关闭沙箱或修改安全设置，只为已授权的云端命令申请正常网络执行权限。不能保证互联网永不故障；它消除了已确认的沙箱误诊原因。

在 Windows PowerShell 的本地仓库根目录执行：

```powershell
Set-Location D:\codex-original
& .\scripts\maintenance\autodl_run.ps1 -Action Check
```

脚本读取 SSH 别名、云端项目、GPU、系统盘/数据盘余量、指定 Python、PyTorch/CUDA 与项目导入路径、`screen` 状态。该命令只报告工程环境，不运行项目预检，不给出 `READY` 或科学判定。没有 GPU 的无卡实例可以用于检查文件；正式 CUDA 工作仍要确认连接的是预期有卡实例。

检查还会确认云端存在 `scripts/maintenance/autodl_watch.py`。后台启动器为每个 `RunId` 绑定独立采集器，每 60 秒记录GPU计算利用率、显存占用、训练Python进程RSS、主机可用内存和生产进度游标到控制目录的 `telemetry.csv`。采集错误写在 `telemetry_status.json`，训练退出状态仍由 `status.json` 单独记录；采集器只读，不恢复或重启训练。任务结束后由启动器停止采集器。

云端首次打开项目后，在 VS Code 运行 `Python: Select Interpreter`，选择 `/root/miniconda3/bin/python`。命令行始终显式调用该 Python，避免非交互 SSH 下 Conda 环境未激活导致解释器变化。

## 后台启动和状态查看

先把**已经确定并审核过的本轮命令**写进本地 UTF-8 命令文件；命令文件不得包含密码、令牌或私钥。脚本只原样运行这个文件，不选择训练入口、检查点、更新数或恢复来源。`RunDir` 必须与命令实际写入的独立生产结果目录一致；操作脚本只记录路径，生产入口仍负责检查和验证它。

```powershell
$commandFile = 'D:\codex-original\outputs\<本轮目录>\runs\<本轮attempt>\approved_command.sh'
$taskName = '<短任务名，只用小写字母、数字、_、->'
$runDir = 'outputs/<YYYYMMDD_本轮名>/runs/<本轮attempt>'

& .\scripts\maintenance\autodl_run.ps1 `
  -Action Start `
  -TaskName $taskName `
  -RunDir $runDir `
  -CommandFile $commandFile
```

保存脚本返回的 `RunId`。任务控制文件、日志和状态写在 `/root/autodl-tmp/autodl-job-control/`，与生产输出目录分开。同名 `TaskName` 已有活动会话时脚本会拒绝再次启动；完成后新 attempt 使用新 `RunId`。启动成功只说明 `screen` 已分离启动，不说明训练成功或测量有效。

查看任务状态和最近日志：

```powershell
& .\scripts\maintenance\autodl_run.ps1 `
  -Action Status `
  -TaskName $taskName `
  -RunId '<Start 返回的 32 位 RunId>' `
  -TailLines 40
```

终态 `COMMAND_EXITED_ZERO` / `COMMAND_EXITED_NONZERO` 只表示 shell 命令退出码；仍需读生产入口的机器报告、日志和预定样本。控制目录或退出记录缺失时状态为 `UNKNOWN`，不得推定成功。脚本不自动选取“最新检查点”；恢复参数由该轮生产入口显式处理。

`Status` 同时显示最近的资源遥测。GPT 对长任务每 60 分钟检查一次状态摘要和新增错误；状态未变化时保持安静，只在异常、重要阶段完成、总完成或需要人工处理时反馈。短诊断只需启动和终局反馈，不建立周期检查。轮询频率不限制训练时长，也不替代云端每分钟采集。

新任务可在生产入口固定参数中登记 checkpoint 策略：每 500 步保存，保留最近两个通过磁盘回读验证的恢复点，并长期保留终点；进度仍每 100 步更新。收到 `SIGINT` 或 `SIGTERM` 时，在当前有效更新结束后立即写入并验证恢复点。只有新写文件通过重新加载检查后才轮换同一轨迹的旧检查点。历史运行没有策略字段时保持原每 100 步、全部保留的行为；恢复仍必须显式指定文件。

固定的 AutoDL 训练任务可显式调用入口的 `--engineering-check-only` 做恢复和保存回读诊断。该模式要求明确的 `--artifact-dir`、`--checkpoint`、`--gradient-audit-dir`、`--reference-dir`、`--evidence-dir` 和本轮专用 `--output-dir`；它不会打开科学attempt、创建/消费READY凭据或启动科学训练。工程Adam新预算最多42次；当前任务保留历史3次，累计上限45次。用 `--engineering-attempt-dir` 指定新编号，用 `--engineering-resume-evidence-dir` 显式恢复同一部分完成的诊断目录；消费者逐阶段重新加载并校验证据，不能仅凭进度数字跳过。输入缺失时先生成文件清单并在CUDA开始前退出；失败阶段保留在原目录，不覆盖旧证据。该入口完成后停止，不自动开始正式训练。

2026-10-13 的 `engineering_attempt_02` 消耗36次新工程Adam更新，累计39/45后停止。保存进程退出及新进程边界状态精确重载通过，但恢复两步后的终点超出预先冻结的同机连续包络；不扩大包络、不称恢复通过。SIGINT/SIGTERM演练和下游保存测量没有执行。继续验收前需要另行核定工程预算，并先定位差异；此诊断不授权科学更新。

在已审核的命令文件中，诊断命令形如：

```bash
/root/miniconda3/bin/python -m scripts.research.lightgen.execute_lightgen_2d_generator_cloud_foreground_dev_multiseed_v1 \
  --artifact-dir outputs/20261009_m0_generator_energy_matched_target_comparison_v1/runs/attempt_03 \
  --checkpoint outputs/20261009_m0_generator_energy_matched_target_comparison_v1/runs/attempt_03/arm_a_step4700.pt \
  --gradient-audit-dir <固定lambda梯度审计attempt> \
  --reference-dir <历史A5400_B5400对照attempt> \
  --evidence-dir outputs/20261012_m0_generator_cloud_foreground_dev_multiseed_v1/runs/attempt_01 \
  --output-dir outputs/20261012_autodl_training_workflow_improvement_v1/runs/attempt_01 \
  --engineering-check-only --microbatch-size 1
```

两个尖括号参数必须替换成已核对的仓库相对路径。恢复演练的 `--evidence-dir` 是只读历史输入，不会让诊断任务续训历史attempt；`--resume-checkpoint` 不能与工程诊断模式组合。

可选 `--microbatch-size {1,2,5,10}` 只改变十行逻辑训练批次内部的光学微批，不改变批次人口、损失定义或更新数。预检会逐候选对比原目标、两臂Generator梯度、Encoder语义梯度及实际更新方向，再进行每个通过候选两次预热和五次计时。只有比微批1的完整逻辑批次中位耗时至少快10%时才建议切换；建议本身不启动训练，也不以显存利用率作为通过标准。

新运行缓存分布先验需要的复数OLS系数和采样比较使用的归一化目标形状向量，不再把全部2000张200×200的SLM目标图复制到恢复缓存。历史缓存格式仍可读取；既有运行和权重不会因新策略被清理或改写。

SSH 或 VS Code 断开不会结束 `screen` 中的进程。AutoDL 实例关机/释放会停止计算；数据盘保留不等于本地备份。不要用关机代替检查运行状态。

## 结果查看和回传

在 VS Code 云端文件树中查看云端生成的报告和固定样本图；长日志可在终端用 `tail -f` 观察，按 `Ctrl+C` 只停止日志查看。完整测量结束后再分析最终报告，不根据中途指标自动延长或改动实验。

回传前按该轮交接列出**精确相对文件路径**，通常先选：

- 可恢复检查点及恢复所需优化器/RNG/采样状态；
- 最终机器报告、训练日志和本轮运行配置/记录；
- 预先指定的固定样本图和诊断读出；
- 若要在本地复现或审核，补充该轮明确要求的数据/模型输入。

大型原始读出按任务需要单独选取。不传整个 `outputs/`。在 PowerShell 中建立本轮新的 `outputs/<版本>/runs/autodl_backup_<名称>` 备份目录并运行：

```powershell
$files = @(
  'report.json',
  'run.log',
  'checkpoint_step_XXXX.pt',
  'fixed_samples.png'
)
$backup = 'D:\codex-original\outputs\<YYYYMMDD_本轮名>\runs\autodl_backup_<名称>'

& .\scripts\maintenance\autodl_pull.ps1 `
  -RemoteRunDir 'outputs/<YYYYMMDD_本轮名>/runs/<本轮attempt>' `
  -Files $files `
  -Destination $backup
```

需要分级归档时，先在云端用 `scripts/maintenance/autodl_review.py package` 从显式证据目录生成 `review` 或 `audit` 包。`review` 包包括报告、验证、人口/输入引用、逐行指标、训练历史、六个终点检查点、必要先验/采样库及三张小图；排除中间检查点、raw读出块和可重建缓存，并标明不能独立完成原始像素审计。`audit` 包另外包括raw读出块。包会记录实际字节数和Windows本地必需的起点文件；回传前，`autodl_pull.ps1` 会拒绝缺失的起点依赖。实例释放或远端清理前，如果后续还需要像素级审计，必须先确认 `audit` 包可读。

`package` 在包目录内生成UTF-8 `review_manifest.json`。先只取回这份小清单，再把 `RemoteRunDir` 指向包目录并用 `-ManifestFile` 回传清单中的精确文件；它与 `-Files` 互斥。脚本把清单经 SSH 标准输入发送，远端预检完整文件路径和大小，再逐文件回传；清单很大时也不会塞进SSH命令行。缺项、依赖缺失或传输失败都留下不完整收据，不会静默缩减文件清单。

```powershell
$packageRun = 'outputs/<YYYYMMDD_本轮名>/runs/<attempt>/review_package'
$manifest = Join-Path $env:TEMP ("autodl-review-manifest-" + [guid]::NewGuid().ToString('N') + '.json')
$backup = 'D:\codex-original\outputs\<YYYYMMDD_本轮名>\runs\autodl_backup_<名称>'
$sshConfig = Join-Path $env:USERPROFILE '.ssh\config'

& scp.exe -F $sshConfig -o BatchMode=yes `
  "autodl-monthly:/root/autodl-tmp/codex-original/$packageRun/review_manifest.json" `
  $manifest

& .\scripts\maintenance\autodl_pull.ps1 `
  -RemoteRunDir $packageRun `
  -ManifestFile $manifest `
  -Destination $backup
```

成功收据只证明清单文件已逐个回传且大小匹配。`review` 包保留了报告和逐行数值，但因不含raw像素块，不能独立复算像素级图像指标；需要此能力时必须选择 `audit` 包并在释放实例前读回核验。工程诊断若在raw测量前失败，应单独清单回传输入核验、阶段结果、预算、性能/恢复报告、RunId和资源CSV；恢复张量及raw读出列入显式audit清单。不存在或未生成的raw不得从清单中悄悄省略后称为完整科学交付。

`scripts/maintenance/autodl_review.py plot` 从保存的逐行指标和训练历史生成三张PNG：六条轨迹的训练遥测折线、与DEV正式判定对应的Q_fg终点柱状图、十类DEV前景误差/输出强度柱状图。工具独立关闭每张画布并验证数值；三张图片总大小限制为2 MiB。已有结果没有资源时序时不补造资源曲线，资源记录保留在CSV供之后分析。

把占位文件名换成该轮实际存在的文件。脚本拒绝通配符、路径越界、受保护命名空间、重复路径和已有备份；先检查远端清单，再逐文件 SCP。它核对返回码和文件大小，在完整备份中写入 `transfer_receipt.json`；传输失败或缺项会在本地保留 `PREFLIGHT_FAILED`、`INCOMPLETE` 或 `PARTIAL` 收据，不标成完整备份，也不删除云端原件。大小用于发现传输不完整，不作为科学准入/恢复锁。

## 固定流程与每轮可变项

固定操作链：

**明确实验 → 环境检查 → 同一生产入口预检 → screen 后台运行 → 查看日志/固定样本 → 完整测量分析 → 关键结果 SCP 回传 → 审阅云端源码变更 → 收尾。**

固定的是路径、解释器、会话保活、日志/状态查看、明确结果目录和回传保护。每轮单独确定的是模型起点、人口/角色、更新数、优化器和调度、批量/精度、架构与损失、科学指标/阈值/样本量，以及是否安排后续训练或模型采用。脚本不会把早期 600/700 步基线写成通用预算。

## 当前科学与工程边界

- A4700仍为研究基线，B5400仍为研究候选。
- 2026-10-11云端冻结端点重放为 `REPLAY_MISMATCH_ENGINEERING_REVIEW_REQUIRED`；Windows保存读出匹配，云端原容差未通过，数值来源尚未定位。保留原容差与原报告，不将跨端差异改写为科学失败或通过。
- 下一轮训练尚未确定科学问题、预算及完整生产预检。此次AutoDL操作链和无模型短任务验收只证明工程控制可用，不表示训练 `READY`，不更改研究判断。

具体训练、指标和停止点仍遵循根 `AGENTS.md`、适用的 scoped `AGENTS.md` 与该轮明确交接。
