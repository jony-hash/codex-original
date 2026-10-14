# AutoDL 连接故障纠正与固定云端执行（2026-10-14）

## 结果与状态

工程状态：`VERIFIED / CONNECTIVITY_RESTORED`。同一条AutoDL环境检查在Codex受限沙箱内退出1（SSH255、主机名解析失败），在沙箱外退出0；后台screen任务、CUDA实际计算、状态查询和精确SCP回传均通过。此前把沙箱DNS/代理错误归因为本机或云服务器网络异常的解释撤回。云服务器无需修改DNS或SSH配置。

## 科学认识

科学判断未改变。本次对照确认当前反复连接报错的原因是执行权限/沙箱网络隔离。沙箱里的DNS、代理查询失败不能据此推出宿主Windows或AutoDL故障。既有G3 FAIL、Gaussian主参考及LP01对照的结论均保持；本次不计算光场、不运行训练或OLS优化。

## 目标缺口

当前连接修复没有未完成项。Internet和云服务未来可能发生其他故障，不能承诺永不报错；已消除本次已确认的沙箱误诊。上一轮逐通道复latent补算仍待执行，连接阻塞已解除；用户本轮要求先解决连接，因此未启动该补算。科学训练或长验证仍按各自已授权任务及有效预检执行。

## 工程完成与验证

证据见[机器报告](../artifacts/autodl_connection_repair_v1.json)、[相同命令对照](../outputs/20261014_autodl_connection_repair_v1/runs/attempt_01/connection_comparison.json)和[精确回传收据](../outputs/20261014_autodl_connection_repair_v1/runs/autodl_backup_attempt_01/transfer_receipt.json)。沙箱外Check重复连接成功；云端为Python3.12.3、PyTorch2.5.1+cu124、RTX4080，项目导入来源正确。screen工程任务退出0，CUDA整数求和观测与期望均为523776；遥测1次、采集失败0次。回传8个文件，全部大小匹配；本地重读probe、退出状态和收据后确认PASS。云端没有源码覆盖、DNS/代理/密钥/host-key修改；只新增本次工程输出。

根AGENTS和AutoDL操作说明已加入实际消费规则：Codex的AutoDL SSH/SCP、Check/Start/Status/Pull显式使用`exec_command.sandbox_permissions=require_escalated`，沿用已授权任务权限。普通用户PowerShell运行方式不变，不自动降低SSH安全校验，也不静默转到本地。

## 决策影响与下一步

后续训练和长数值验证固定在`autodl-monthly:/root/autodl-tmp/codex-original`，解释器为`/root/miniconda3/bin/python`；Windows本地负责源码、准备、审阅和备份。此用户选择覆盖此前本地训练默认说明。已有科学授权保留，但本轮仅完成连接工程验收并停止；正式训练更新、光学前向、数据集读取和评价器调用均为0。先前OLS后处理可在该通道继续，不能把连接正常当作G3通过。
