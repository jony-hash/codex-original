# 2026-10-15 D/E/AutoDL 存储整理

**范围：** D盘继续作为工作项目盘，E盘用于备份，AutoDL继续作为计算端。本轮只处理磁盘文件；未关闭应用、调整RAM、修改环境或启动训练/实验。科学判断和训练授权未改变。

**当前状态：** 本地及AutoDL备份、内容核验、恢复演练和按逐文件清单清理均已完成。末次容量读数：D盘空闲约383.34 GiB，E盘约192.44 GiB，AutoDL约47.12 GiB。

## 结果与状态

D盘保留工作项目；E盘保留本地项目镜像（ACL限制例外见下）、历史归档复用映射、Git bundle、AutoDL分片快照及差异补充包。已清理本地6,930个PT和云端5,240个候选文件；云端活动任务检查通过，未停止应用或任务。初次云端最终核验发现的38个新增路径、一个持续变化的SQLite监控库和符号链接映射缺项均已补备份或补正映射，并通过复核。

## 科学认识

科学判断未改变。本轮只处理存储、备份与恢复能力，不执行训练或实验，也不把存储核验结果解释为科学证据。

## 目标缺口

本事项目标是可恢复地备份并释放空间；项目的科学证据缺口和停止点没有因此改变，继续以 `docs/CURRENT_STATUS.md` 与当前交接为准。

## 工程完成与验证

旧归档成员、D盘待清理PT、云端候选文件和新增云端文件均在删除/交付前做内容核验。云端清理时每个文件均在远端复核大小与SHA-256后按精确路径删除；本地历史PT、新PT及云端NPZ恢复演练均通过。详细清单、哈希和恢复映射保存在 `E:\lg261015`。

## 决策影响与下一步

D盘继续工作，E盘作为离线备份接入点，AutoDL保留计算端。需要历史重放时先接入E盘，再按恢复映射恢复任务实际引用的大文件。没有自动后继实验或训练授权。

## 备份位置与恢复映射

本轮新增内容放在 `E:\lg261015`，不覆盖旧归档。`E:\LDM-Archive\codex-original` 保持不变。旧归档163个tar成员共33,117个文件、240,556,414,616字节（224.04 GiB），已逐成员与D盘文件逐字节比较，差异、缺失和读取错误均为0，故这些文件直接复用旧归档。

本地项目最初清单为75,460个文件、13,553个目录和2个指向C盘运行库的目录联接。旧归档覆盖33,117个文件，其余42,343个文件按D盘项目相对路径增量保存到 `E:\lg261015\l`；复制期间对源文件做稳定性检查，并对备份做SHA-256回读。随后发现清单生成后新增104个文件，另有55个路径元数据变化，已将159个新增/变化文件补存并验证，共786,425,353字节（0.7324 GiB）。其中5个源码/测试文件或字节码与初始清单不一致，E盘保存的是复核时稳定的当前版本；差异在 `local_inventory_delta.jsonl`、`local_inventory_reconciliation_verified.jsonl` 和 `local_map_final.jsonl` 中留有记录。

E盘另有Git可达历史bundle：`E:\lg261015\codex-original-history.bundle`，大小88,949,713字节；`git bundle verify`通过，包含68个引用，当前分支HEAD为`043e2544594642669914355d35040124b64459b7`。工作区未提交和未跟踪文件通过本地增量镜像保存，没有创建提交。

逐路径恢复映射为 `E:\lg261015\local_recovery_map.jsonl`。每一行记录D盘原路径、源状态、备份类型和位置、归档成员、校验状态及恢复方式。被删除PT的逐项删除结果在 `local_cleanup_result.jsonl`；原始候选与保护名单分别在 `local_cleanup_candidates.jsonl` 和 `local_cleanup_protected.jsonl`。清单后出现的文件和变化记录在 `local_inventory_delta.jsonl`。旧归档完整性摘要在 `archive_reuse_preflight.json`。

**已知本地备份缺口：** 两张联系表受NTFS ACL限制，无法读取，仍留在D盘，未改ACL：

- `outputs/20260928_m0_bogt_ols_scale_gradient_rescue_v1/runs/attempt_01/end_contact_sheet.png`
- `outputs/20260928_m0_bogt_ols_scale_gradient_rescue_v1/runs/attempt_01/start_contact_sheet.png`

最终递归清单读取时另有85个目录拒绝枚举，具体目录见 `local_inventory_delta.jsonl`；无本地PT清理候选位于这些目录下。两处C盘`node_modules`目录联接只记录了链接目标，不遍历或复制C盘运行库。初始映射中有9条Codex临时Git路径在后续检查时已不在D盘；本地映射保留其初始记录，Git bundle保存了最终可达引用。

## 本地清理结果

清理候选只来自 `outputs` 内`.pt`文件：版本目录日期早于2026-09-01，或之后单文件至少256 MiB。`outputs/final_pt`、35个受保护PT及依据当前状态/交接/配置识别的依赖均未删除。新出现的2026-10-15输出未进入初始清理计划，已补入备份并保留在D盘。

6,930个候选（282,397,965,565字节，263.0036 GiB）均在删除前与对应旧tar成员或E盘增量副本逐字节比较，核验通过后才删除；失败或来源变化时脚本会保留原件。本批实际增加D盘空闲282,419,388,416字节（263.0236 GiB）：清理前空闲129,196,261,376字节（约120.36 GiB），清理后411,615,649,792字节（约383.35 GiB）。删除清单中没有失败或保留项。

清理后复核确认35个保护PT、`outputs/final_pt`的2个文件以及最终映射中的13,352个JSON/JSONL/JSON5文件仍在原位置。现有D盘环境以禁止生成字节码的方式导入`ldm_project`通过；未运行训练或全量测试。历史PT恢复演练和增量PT恢复演练都与D盘原件逐字节相同，详情见 `local_restore_rehearsal.json`。清理与核验摘要在 `local_cleanup_result.json` 和 `local_post_cleanup_verification.json`。

云端清理保护引用中有5个文件在D盘存在但最初未进入E盘映射；已从D盘只读暂存到E盘，逐项核对大小和SHA-256，补充记录在 `local_protected_gap_recovery.jsonl` 与 `local_protected_gap_recovery_summary.json`。这些源文件仍在D盘，未调整ACL。

## AutoDL 备份与清理

云端源根为 `/root/autodl-tmp`。完整快照由 `cloud_snapshot_part00.tar` 至 `cloud_snapshot_part06.tar` 组成，覆盖隐藏文件、`codex-original`源码与outputs、`autodl-job-control`、`incoming`、`migration_records`及同级数据。初始清单和恢复映射均为12,454个路径（1,531目录、10,922文件、1个符号链接），没有重复、漏项或类型错误。一次约45 GB的连续SSH传输中断后，保留了结构完整的part00，并将剩余内容重建为6个分片；七个分片均完成成员读取和结构验证。快照过程中12个活动文件大小发生变化，已通过远端SHA-256与云端当前版本逐项对账，记录在 `cloud_snapshot_part01_mutation_reconciliation.json` 和 `cloud_snapshot_part06_mutation_reconciliation.json`。

首次完整快照后的实时核验发现38个新增路径（4个目录、34个文件，主要是2026-10-15 quickcheck输出和任务控制记录）以及持续变化的 `.autodl/autopanel.monitor.db`。新增34个文件共370,227,194字节，已存入 `cloud_snapshot_delta_20261015.tar`；打包前后哈希稳定，E盘归档内35个普通文件成员（包括数据库快照）全部逐项复核，34个新增文件在云端再次哈希也全部匹配。SQLite库通过在线备份接口生成3,518,464字节的一致快照，`PRAGMA quick_check`为`ok`。增量包传输前后SHA-256一致。恢复叠加映射为 `cloud_snapshot_delta_recovery_map.jsonl`，对账摘要为 `cloud_snapshot_final_reconciliation_summary.json`。

初始恢复映射曾漏记part06中一个测试临时符号链接的目标；直接读取归档确认其目标与云端实时链接一致，已在增量恢复映射中补正，归档无需重传。最终云端实时清单为7,251个路径，即原始快照删除后保留的7,213个路径加38个新增路径；缺失、类型不匹配、大小不匹配和删除路径重现均为0。原始JSON保护清单1,140项全部仍在云端。候选保护审计覆盖55个路径：4个原先存在于云端的保护文件均保留，另外51个在本次操作前就不在云端，其中5个已补入D盘到E盘的恢复映射。

最终删除清单包含5,240个历史PT、NPZ、原始读出或重复中转包，共42,648,622,829字节；另有152个候选（2,183,096,866字节）因近期输出或用途不确定而保留。每个删除文件均先与E盘归档做SHA-256比对，并在远端删除前重新核对大小、哈希和文件状态；5,240项全部匹配后才按明确路径删除。逐文件计划、哈希和执行结果分别见 `cloud_cleanup_delete_plan.jsonl`、`cloud_cleanup_backup_hashes.jsonl`、`cloud_cleanup_execution_attempt03_remote_output.jsonl` 及其摘要。云端NPZ恢复演练文件为 `cloud_npz_restore_rehearsal.json`：恢复文件533,287,769字节，SHA-256匹配，内层36个ZIP成员CRC通过。

AutoDL清理前空闲7,892,647,936字节，完成后为50,552,303,616字节，文件系统实增空闲42,659,655,680字节；文件系统总量53,687,091,200字节，使用3,134,787,584字节。检查时无screen任务、项目训练/诊断进程或GPU进程；Jupyter-Lab和TensorBoard保持运行。临时传输包在E盘归档核验完成后按精确路径移除。云端原始清理计划有5,392个候选；最终因保护和用途复核，仅执行上述5,240项。

## 恢复方式与回退

本地增量文件按恢复映射从 `E:\lg261015\l\<项目相对路径>` 复制回 `D:\codex-original\<项目相对路径>`。旧归档成员按映射中的归档文件和成员路径解包，例如：

```powershell
tar -xf 'E:\LDM-Archive\codex-original\outputs\archive_0141.tar' `
  -C 'D:\codex-original' `
  'outputs/20260818_v2/runs/20260818_172608_790649_research_fiber_ols_zero_shot/mmf_capacity_matched_nominal_ols_diagnostics.pt'
```

上例仅用于恢复演练。历史重放若依赖本次移出的模型/优化器/RNG/采样PT，应先接入E盘，再按 `local_recovery_map.jsonl` 还原该任务实际引用的路径；不必把整个旧归档解回D盘。Git提交历史可由bundle验证或恢复，未提交文件从`l`镜像恢复。AutoDL文件完成打包后会使用tar成员映射恢复到E盘测试目录；如要回传云端，再按映射恢复到其原始云端路径并校验哈希。

本地清理可通过上述恢复位置回退；不移动D盘项目目录、不更改Conda环境、AutoDL任务或Git提交。需要恢复云端快照时，先在E盘恢复目录解出part00–part06，再按映射把 `cloud_snapshot_delta_20261015.tar` 叠加到其上；该增量包含快照后新增路径和更新后的SQLite一致快照。清理后的历史云端PT/NPZ按 `cloud_snapshot_recovery_map.jsonl` 和补充映射恢复到原始云端相对路径。重放依赖本轮移出的历史检查点、优化器、RNG、采样库或数据输入时，必须先接入E盘并按本地或云端恢复映射恢复该任务明确引用的文件。旧归档manifest及历史路径未改写。
