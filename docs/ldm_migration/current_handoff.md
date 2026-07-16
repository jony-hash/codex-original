# 当前主线交接：v65

更新时间：2026-07-15。此文件供新对话快速恢复当前主线；带日期的完整证据在 [Experiment History](experiment_history.md)，稳定架构合同在 [D2NN 单次曝光蒸馏主线](single_shot_ldm_distillation.md)。

## 先读的结论

- 当前所有 v49-v65 结果都是 `diagnostic-only`，且 `accepted_summary_written=false`。没有可 promotion 的学生 checkpoint。
- 人眼硬 gate 只看最终 `physical_camera_plane` 的 CMOS intensity。v56 的正式 class-balanced sample（0-9 每类两张）仍为同源圆斑/纹理，不可读；这是 strict provenance 有效条件下的模型失败，不是 harness failure。
- 不要重新使用 v47/v51/v53 checkpoint：它们与现行的 identity OLS、1.50 mm geometry 或完整前端 4f relay 合同不兼容。
- v61 的原生采样离线 ridge 上界已从冻结 v60 camera intensity 重建出 held-out teacher target，但它是 900 例训练/验证后的软件读出；最终 `physical_camera_plane` 对人眼仍不可读。它不改变上述硬 gate，也不授权训练。
- 在下一决策门前，不训练、不延长 epoch、不叠加 loss、不做 spacing/padding/pupil sweep，也不修改正式硬件合同。

代码已完成行为保持的领域重构：`optics / student / teacher / application / diagnostics` 为正式目录，stage 通过 `StageRequest` 懒构造依赖；旧 import 仍解析到同一实现。此结构状态不改变下列 v56–v59 科学结论。

## 现行正式理想数字孪生合同

```text
laser -> M1 -> BS -> amplitude SLM -> BS -> M3
      -> coherent 4f relay -> encoder D2NN (2 layers)
      -> complex identity OLS
      -> decoder D2NN (2 layers)
      -> finite-conjugate coherent relay -> CMOS intensity
```

- `lambda=532 nm`；无 fiber。OLS 是严格复线性恒等映射 `Eout=Ein`。
- SLM/CMOS：`200x200 @2.5 um`，500 um FOV。
- Encoder/decoder：各 `100x100 @2 um`，200 um FOV，各两层，phase-only，spacing `1.50 mm`。
- 前端完整 FOV 相干 4f 缩小成像：`M=-0.4`、`f1=223.214 mm`、`f2=89.286 mm`、最低 pupil `23.75 mm`；禁止 central aperture crop 或 generic complex resize。
- 后端有限共轭成像：`M=-2.5`、`u=100 mm`、`f=71.4285714 mm`、`v=250 mm`、最低 pupil `26.60 mm`。
- 严格光场 artifact 必须从同一传播复场的 `real/imag` 现场计算 `abs(E)`/`angle(E)`；camera intensity relay 不得伪造 phase。

v55 structural preflight 通过；原 v55.1 reference 仅证明生产算子与同构 Collins 实现自洽，已降级为非独立证据。v55.2 的低分辨率实际尺寸 direct-RS 参考也因采样不足而无效。v58.2 CUDA run 证明 DFT/NUDFT 可执行，但其 `128→64` sampling、硬编码 Airy error 与单段 component reference 不能签发正式 gate。现行 v58.3 report 是完整 `200→100` FOV、闭式 B=0 场、实测 pupil PSF 与故障注入独立性审计的正式前向前提。

- `outputs/reports/v55_coherent_relay_cuda_preflight.json`
- `outputs/reports/v55_1_coherent_relay_reference_cuda.json`（历史自洽检查）

这些是理想数字孪生与设计 pupil 下限的结构/历史自洽检查，不是独立物理参考或硬件实测。

v49-v54 的 mask 归因、严格复场修复和旧 central-aperture 失败定位均已降为历史证据；完整细节见 [Experiment History](experiment_history.md)。v60 是 v56 的同配置复制训练，未改善人工可读性。v61 稳健审计未确认稳定的局部首塌缩；新的 native dual-channel ridge 上界确认了软件可解码性，而非 CMOS 人眼可读性。v63 进一步确认 code 因子不是在正式光路被局部抹除；下一门只能隔离原生、不可读的表示与人眼可读性的差异，不得训练。

## v56-v59 事实与证据

v56 best checkpoint（epoch 5）：

```text
outputs/20260714_v56/runs/20260714_204958_697374_train_v56_coherent_relay_from_scratch/checkpoints/one_shot_distill_best.pt
```

训练从 epoch 1 到 5：validation loss `0.2715 -> 0.2276`，paired corr `0.4913 -> 0.5870`，paired dice `0.5704 -> 0.6351`；但 diversity ratio `0.00146 -> 0.00095`，cross-class ratio `0.000138 -> 0.000103`。

正式 eval：corr `0.5789`，dice `0.6298`，diversity ratio `0.000968`，strict readability failed：

```text
outputs/20260714_v56/runs/20260714_210206_240133_train_v56_coherent_relay_eval/artifacts/one_shot_distill_eval_metrics.json
```

正式 sample/provenance：

```text
outputs/20260714_v56/runs/20260714_210416_032022_train_v56_coherent_relay_sample/
```

v57 corrected report：

```text
outputs/reports/v57_first_collapse_complex_field.json
```

v57 以 SLM 的跨类中位距离减同类中位距离为 100% 基准，得到：前端 relay 约 47%，encoder/OLS/decoder/chip 的 complex separation 约 39%-49%，camera intensity 约 11.7%。OLS 前后连续。没有任何 plane 满足“至少两类指标低于 10% 且持续到下游”的局部首塌缩门。

v58 aperture-throughput upper bound 已完成，且人工并排复核 baseline 与 upper-bound 的最终 CMOS 都是同源圆斑/纹理、不可读。它在同一 v56 checkpoint、相同输入/relay、零训练下移除了 D2NN 层间 finite crop；chip 能量从 `126.18` 增至 `17232.99`，但 camera field 能量仅从 `66.27` 增至 `69.78`，CMOS separation gain 只有 amplitude `1.0227x`、complex `1.0173x`、intensity `1.0101x`。因此 finite crop 是高吞吐损失，但不是类别/图像塌缩的充分根因：

```text
aperture-upper-bound-no-separation-recovery
manual-review-required
diagnostic-only
accepted_summary_written=false
```

v58 报告与 tensor：

- `outputs/reports/v58_aperture_throughput_upper_bound.json`
- `outputs/reports/v58_aperture_throughput_upper_bound.pt`

v59 upstream broad-isolation 已在同一正式合同与 v56 checkpoint 上完成，验证了 100 张 readable validation（0-9 每类 10 张）、`readability_strict_split`、identity OLS、正式 relay geometry 与参数零变化。teacher target 的 prototype CV 为 `1.00`，permutation `p=0.00775`；baseline SLM command 与前端 4f 输出的 prototype CV 均为 `0.94`，前端相对 SLM 的 complex separation 为 `41.0%`，SLM 相对 teacher target 的 complex separation 仅 `0.6787%`。SLM 类间频谱约 `80.2%` 在形式通带外，但 4f 输出仍保留显著类别 prototype signal，故未满足 bandwidth mismatch 门。label-shift 与 code-rotation 对最终 target MSE 的相对变化分别为 `0.155%` 与 `0.0036%`；最终 CMOS 仍不可读。

v59 corrected report 与 artifacts：

- `outputs/reports/v59_upstream_separation_isolation.json`
- `outputs/reports/v59_upstream_separation_isolation.pt`
- `outputs/reports/v59_upstream_separation_isolation_preview.png`

当前 v59 决策为：

```text
objective-representation-mismatch-candidate
manual-review-required
diagnostic-only
accepted_summary_written=false
```

下一门是离线 decodability upper bound；不得继续 decoder/loss 微调、延长 v56 epoch、sweep aperture/spacing/pupil，或 promotion v56。

每层 numerical propagation 能量比约 `0.999999`；但 finite aperture crop retention 是 encoder L1 `25.70%`、encoder L2 `27.65%`、decoder L1 `30.83%`、decoder L2 `33.42%`，累计 `0.7322%`。其类别差异不显著，故当前状态是：

```text
aperture-throughput-limited-no-localized-collapse
manual-review-required
diagnostic-only
accepted_summary_written=false
```

## v59 下一决策门

v59 tool：`tools/diagnose_v59_upstream_separation_isolation.py`。它只做固定 validation batch 的 baseline、label derangement、within-class code rotation forward，并将类间/类内分离、permutation null、target error、label/code sensitivity 与 formal pupil spectral audit 写入报告。当前报告指向 offline decodability upper bound；其仍是 `diagnostic-only`，不改变正式合同。

## v60-v61 当前证据与下一门

v60 使用与 v56 相同的 readable 数据、seed `1234`、5 epoch 和正式 `coherent_imaging_4f/free_space/identity OLS` 合同；它只是复制训练，不是新物理架构。相较 v56，eval corr/dice 为 `0.5733/0.6235`（v56 为 `0.5789/0.6298`），sample corr 为 `0.4914`（v56 为 `0.4945`），最终 CMOS 仍为跨类别同源圆斑/散斑。

v61 robust audit 使用冻结 v60 best checkpoint 和 readable `800/100/100` train/val/test，参数 hash 前后相同；以 32×32 诊断网格、2,000 次固定大小 pair bootstrap 和 permutation null 复核全部正式 planes。当前报告为 `no-robust-localized-collapse`，因此 v60 的单次 20 样本、`0.0981` camera 阈值边缘报告不能单独升级为 camera 首崩溃结论。

代表证据与工具：

- `outputs/20260715_v61/runs/robust_collapse/v61_robust_collapse.json`
- `outputs/20260715_v61/runs/robust_collapse/v61_frozen_optical_cache.pt`
- `tools/diagnose_v61_robust_collapse.py`
- `tools/diagnose_v61_offline_decodability.py`

早期的 `diagnose_v61_offline_decodability.py` 是带降采样/PCA 与额外对照的历史软件 readout 研究；其可读结果不能替代本轮的原生采样门，也不能进入正式单曝光推理链。

v61 native dual-channel upper bound 冻结 v60 epoch-5 checkpoint，沿严格正式前向读取完整 readable `800/100/100` split；没有 optimizer、无参数变化，且逐批验证 `physical_camera_intensity_raw == |E_camera|²` 与 detector-normalized `physical_camera_plane` 语义。四个 held-out linear ridge probe 均只在 train 拟合、在 validation 从 `{1e-4,1e-3,1e-2,1e-1,1}` 选择、再以 train+val 重拟合并只评估 test；没有 resize、PCA、神经网络或复场 bundle 输出。

| Native probe | Test corr | Dice | Prototype label acc. |
| --- | ---: | ---: | ---: |
| chip intensity | 0.9453 | 0.8829 | 1.00 |
| chip complex oracle | 0.9456 | 0.8835 | 1.00 |
| camera intensity | 0.9452 | 0.8829 | 1.00 |
| camera complex oracle | 0.9456 | 0.8835 | 1.00 |

因此该预注册门为 `camera-intensity-decodable`：camera complex oracle 没有带来实质增益，且 chip/camera intensity 都达标，故当前证据不支持“仅 phase-to-intensity”或“仅后端 relay”信息失配。它仅说明有一个离线线性软件 readout 可以从不可读 CMOS intensity 的细微条件相关差异恢复 teacher target；它不代表 raw CMOS 对人眼可读、不是光学末端生成，也不允许 promotion 或继续训练。下一门应为零训练的 native conditional-control audit：预注册 label-only、code-only、within-class target permutation 与逐实例残差信号，以判定该上界是否只是类别模板。

新证据：

- `tools/diagnose_v61_offline_decodability_upper_bound.py`
- `outputs/reports/v61_offline_decodability_upper_bound.json`
- `outputs/reports/v61_offline_decodability_upper_bound.pt`
- `outputs/reports/v61_offline_decodability_upper_bound_preview.png`

## v62 解释修正与 v63 code-conditioning provenance audit

v62 在冻结 v60 epoch-5 checkpoint、正式 v56 config、同一 `readability_strict_split` 数据和 v58.3 relay gate 上完成一次零训练 native conditional-control audit。输入验证通过：train/val/test 为 `800/100/100`、每类分别 `80/10/10`，identity OLS、正式 4f/后端 geometry 和 `physical_camera_intensity_raw == |E_camera|²` 语义均保持；模型参数 hash 前后无变化，`optimizer_created=false`。

对 held-out test 做了三组原生 forward：baseline；固定一个 readable-train code、仅变化 requested label 的 `label-only`；保留 requested code、将 label 固定为 0 的 `code-only`。同时复用 v61 已封存、未重新拟合的 camera-intensity ridge prediction，进行 within-class target 一位循环置换和 leave-one-out class-template residual 对照。所有反事实均为 `diagnostic-only`。

v62 的仍然有效的事实：

- label-only prototype CV accuracy `1.00`，permutation `p=0.00775`；其类内 residual energy `3.14e-15`，说明类别模板几乎完全由 label 条件决定。
- baseline prototype CV accuracy `0.94`，permutation `p=0.00775`；code-only 用原始 readable-test labels 分组的 accuracy `0.23`，`p=0.00775`，类内 residual energy 仅 `5.09e-5`。
- baseline 与 label-only camera 的归一化响应为 `1.040%`；baseline 与 fixed-label code-only 的响应为 `0.389%`。
- v61 camera-intensity offline prediction 与配对 teacher target 的 corr `0.9452`；within-class target rotation 后为 `0.8909`，paired delta `0.0544`，95% bootstrap CI `[0.0469, 0.0628]`。
- 去除 leave-one-out 类别模板后的 residual corr 为 `0.4931`，同类 target rotation 后为 `-0.0890`；paired residual delta `0.5821`，95% CI `[0.5321, 0.6325]`。这证明离线读出中存在逐实例残差信号，但不等于 raw CMOS 人眼可读。

v63 发现上述 v62 `0.23` 不能用作学生的 code 类别学习证据：全局 15k IID 导出 code 的 prototype CV 是 `0.0971`（null `p=0.9412`），但 readable 筛选后的 train/val/test 分别为 `0.3225/0.1700/0.2200`，test `p=0.00775`。因此这是 target readability 筛选引入的 code-label selection confound；它不能推翻已封存的 paired-residual 对照，也绝不授权训练。v62 的 `baseline→label-only 1.040%` 实际是替换 code、保持 requested label 的反事实响应，不能当作 label 响应；固定 `2%` 绝对响应门同样不再用于 code 根因判定。

v63 用全局 `code_seed=1234` 精确重放 15k code，并以 `code_projection_seed=2026` 精确重算每个 readable split 的 `teacher_initial_noise`；所有 code/label/noise replay 都相等。冻结 v60 epoch-5 checkpoint 后，对 100 held-out test code 与全部 10 labels 的 `100×10` 平衡交叉网格沿未改动正式链路前向，无 resize、optimizer、readout refit 或参数变化；camera raw intensity 与 detector normalization 语义也逐批通过。

| Plane | Code factor | Label factor | Interaction |
| --- | ---: | ---: | ---: |
| SLM command | 80.62% | 16.07% | 3.31% |
| pre-encoder intensity | 80.51% | 16.13% | 3.35% |
| chip intensity | 79.52% | 17.71% | 2.77% |
| final CMOS intensity | 80.39% | 15.55% | 4.06% |

预注册决策同时满足 `generator-code-dominant`（SLM code factor ≥25% 且高于 label）与 `native-code-conditioned-subvisual-representation-confirmed`（camera 保留 code factor，v62 paired residual 有效）；不满足任何“SLM 的 10% 以下而 label 持续保留”的局部 code 因子衰减门。真实 raw CMOS 的人工不可读结论不变，故这不是 promotion 或训练授权。下一门只允许设计离线/诊断性的 representation-readability isolation，明确比较 code/label 因子可保留性与最终 CMOS 的人眼可读性；不得回到 decoder/loss 微调或改变正式光学合同。

v62 工具与工件：

- `tools/diagnose_v62_native_conditional_control_audit.py`
- `outputs/reports/v62_native_conditional_control_audit.json`
- `outputs/reports/v62_native_conditional_control_audit.pt`
- `outputs/reports/v62_native_conditional_control_audit_preview.png`

v63 工具与工件：

- `tools/diagnose_v63_code_conditioning_provenance_audit.py`
- `outputs/reports/v63_code_conditioning_provenance_audit.json`
- `outputs/reports/v63_code_conditioning_provenance_audit.pt`
- `outputs/reports/v63_code_conditioning_provenance_audit_preview.png`

## v64 semantic-band carriage audit

v64 固定 v60 epoch-5 checkpoint、v56 正式 config、v58.3 relay 和 readable `800/100/100` split，沿未改动正式链路提取 native `physical_camera_plane`。15k code/noise provenance、v61/v63 labels/teacher targets、identity OLS、geometry、camera `|E|²` 语义和参数快照均有效；`parameter_update_count=0`、`optimizer_created=false`、未保存正式复场 cache。

以 camera `200x200 @ 2.5 µm` 的物理 DFT lattice 定义 `0–20 cycles/mm` semantic band、`20–80` mid band 和 `>80` high band。train teacher target 在 semantic band 内含 `95.96%` 中心化空间能量，超过预注册 `90%` cutoff；但 v63 camera code-main 仅有 `20.16%` 位于 semantic band，`67.47%` 在 mid、`12.37%` 在 high。label-main 频带份额为 `18.24%/69.56%/12.20%`。

独立 band-only held-out ridge probes（不重拟合 v61 raw probe）结果如下：

| CMOS feature | Corr | Dice | Paired rotation drop (95% CI) |
| --- | ---: | ---: | ---: |
| semantic `0–20` | `0.9421` | `0.8782` | `0.0547 [0.0474, 0.0626]` |
| non-semantic `>20` | `0.9454` | `0.8832` | `0.0545 [0.0469, 0.0625]` |
| frozen v61 raw | `0.9452` | `0.8829` | reference |

两个频带都保留 held-out paired information，且 semantic probe 达到 raw baseline 的 90% retention；但 code-main 没有在 semantic band 占多数。因此预注册决策为：

```text
distributed-subvisual-carrier
manual-review-required
diagnostic-only
accepted_summary_written=false
```

这说明 paired representation 不是单一低频或高频载波，而是分布在多个频带；它仍然不能解释 raw CMOS 人眼不可读，也不授权修改 detector、display、generator、loss、D2NN、relay 或启动训练。下一门只允许做跨频带表示归因与固定显示/人工审查关联，继续停止训练和 checkpoint promotion。

v64 preview 人工复核：teacher 行可读；raw CMOS 仍为同源颗粒纹理；semantic 行主要是几乎相同的低频模板；non-semantic 行主要是颗粒纹理。故 raw CMOS 人眼 gate 仍失败。

v64 工具与工件：

- `tools/diagnose_v64_semantic_band_carriage_audit.py`
- `outputs/reports/v64_semantic_band_carriage_audit.json`
- `outputs/reports/v64_semantic_band_carriage_audit.pt`
- `outputs/reports/v64_semantic_band_carriage_audit_preview.png`

## 第二轮解耦状态

本轮代码重构不改变 v56–v63 科学结论，也没有启动训练。当前正式路径的 phase-only D2NN、SLM transfer 和 identity OLS canonical 实现在 `optics/`；正式 Student 由 `FormalOneShotOpticalGenerator` 构造，旧 `OneShotOpticalGenerator`、`student/layers.py` 和根目录 one-shot imports 仅保留 compatibility/diagnostic 行为。checkpoint 合同位于 `student/checkpoints.py`，clean-readable target 位于中立 `distillation/targets.py`，stage student model 由 `application/model_factory.py` 按 physical contract 延迟构造。

截至本轮已验证：v62/v63 基线完整测试 `478 passed`、`148 subtests passed`；formal Student/Optics 定向测试通过；冻结 v60 checkpoint 在 CUDA 上与 `/tmp/second_decoupling_baseline.pt` 的 96 个输出字段和 15 个 provenance 光场平面 bitwise 相同。随后 objective composer 已将 `camera_recon_loss` 固定为 formal term，其余项按显式 experimental registry 叠加；checkpoint 实现已落到 `student/checkpoints.py`，旧 model 私有入口仅作兼容转发。新增 composer/边界测试后完整测试为 `481 passed`、`148 subtests passed`。完整 stage context、config schema 和超大测试文件的进一步机械拆分仍需在下一等价门之后继续，不能与科学训练混合。

第四轮已将正式 Student 实现独立到 `student/formal_model.py`，兼容模型归入 `student/experimental/model.py`，`student/model.py` 仅保留公开 facade；冻结 v60 checkpoint 在 CUDA 上与 `/tmp/second_decoupling_baseline.pt` 的 96 个输出字段 bitwise 相同。配置 dataclass、YAML loading/normalization 和 validators 已分别落到 `configuration/schema.py`、`loading.py`、`validation.py`，旧 `legacy.py`/`config.py` 仍保持对象和错误契约。teacher distill 算法主体已移到 `teacher/common.py`，旧入口保留 prototype builder monkeypatch 注入点；stage runtime footprint 位于 `application/stage_runtimes.py`。完整回归为 `490 passed`、`148 subtests passed`、1607 个既有 warning；本轮不训练、不改变 v56–v63 科学结论。超大测试文件机械拆分和逐 runtime 物化的进一步收敛仍延后，不能把 footprint 登记误写成已完全替换 StageContext。

第五轮已完成正式 Student 固定物理链收口：`FormalOneShotOpticalGenerator` 不再包含 diagnostic intervention，兼容路径由 `student/experimental/model.py` 按配置选择；输出 contract 与 optical provenance 组装独立到 `student/output_contract.py`。`application/stage_runtimes.py` 现在负责实际依赖物化，`StageContext` 仅为兼容适配器；data-only、student、teacher 阶段的模型集合会在构造后再次校验。正式 camera reconstruction term 已迁入 `student/objective_terms/formal.py`，旧 `one_shot_loss` 继续保持签名、返回字段和浮点加法顺序。冻结 v60 checkpoint 的 CUDA 96-output/9-state-key 比较通过，完整回归为 `492 passed`、`148 subtests passed`、1607 个既有 warning；本轮仍不训练、不改变 v56–v63 科学结论。超大测试机械拆分及其余实验 objective term 的移动留待独立提交。

第六轮完成 Teacher、Objective 与测试规模收口：teacher workflow 已分别迁入 `teacher/export.py`、`label_audit.py`、`filter.py`、`readable.py`，`teacher/common.py` 降为兼容 facade；prototype builder monkeypatch 改为显式依赖注入。实验 objective 已按 readability、semantic、topology、diversity、residual 分组到 `student/objective_terms/`，`student/objectives.py` 保留 loss 编排。原两份超大测试已按 student/application/diagnostics 分类，公共 fixture 集中到 `tests/tests_support.py`，旧测试文件保留兼容 smoke。该轮中间分类回归为 `282 passed`；最终完整仓库回归为 `498 passed`、`148 subtests passed`、1607 个既有 warning。未训练、不改变 v56–v63 科学结论。

第七轮完成最终边界校正：根包模型导出改为惰性属性，data-only stage 不再因兼容 root import 加载 VAE、DDPM 或 Student；`stages.py` 与 `stage_runtimes.py` 按 footprint 延迟绑定模型、loader 和 hooks。D2NN tensor/metric primitive 归入中立 `evaluation/d2nn*.py`，`student` 不再导入 `diagnostics`；旧 diagnostics/root 路径继续作为兼容 facade。Teacher scoped workflow 和 objective terms 改为显式依赖，新增 Teacher 边界与 subprocess import 测试。最终完整回归、静态检查和工作树状态以本节末尾的最新验收记录为准；不训练、不改变 v56–v63 科学结论。

## 已完成验证

第七轮源码收尾验收：`503 passed`、`148 subtests passed`、1607 个既有 warning；`ruff check src tests *.py`、`python -m compileall src tests train.py`、`git diff --check` 和 subprocess lazy-import boundary tests 均通过。未运行新 CUDA 训练，未修改或提交 `outputs/`、checkpoint、数据集或缓存。

v58.3 targeted/CUDA preflight、v61 robust audit、v61 native dual-channel upper bound、v62 conditional-control audit、v63 crossed-factor audit 与 v64 semantic-band audit 已完成；v64 CUDA harness 有效，参数更新数为 0，所有报告均为 `diagnostic-only`。本次代码的完整验证命令及结果见当前工作区测试输出；所有输出和 cache 均位于 `outputs/`，不加入 Git；迁移时不得用 reset/checkout 清理用户改动。

### v65：common-mode 与 readout-locality attribution audit（2026-07-15）

- 固定 v60 epoch-5 checkpoint、v56 正式 config、v58.3 relay、readable `800/100/100` split 及 v61/v63/v64 工件，执行一次零训练 CUDA audit。输入 provenance、identity OLS、formal geometry/field chain、camera `|E|²`/detector normalization 和参数快照均通过；`parameter_update_count=0`、`optimizer_created=false`，临时 native `200x200 @2.5 µm` CMOS memmap 在诊断后删除。所有结果为 `diagnostic-only`，未改 formal inference 或硬件合同。
- 训练集唯一 common mean 的 RMS 为 `0.230195`，训练 residual `abs` 99.5 percentile 为 `0.0071021`，固定全局 gain 为 `63.3615`。无监督 common-mode display 的 test corr/dice/prototype accuracy 仅 `0.04295/0.33365/0.46`，class-balanced preview 仍不可读；共同模板/显示动态范围不能单独解释为可读恢复。
- 冻结 v61 raw camera reference 为 corr/dice `0.94523/0.88293`。held-out observer ladder 的 test 结果为：pointwise affine `0.68988/0.68732`；shared local convolution `3x3/7x7/15x15/31x31` 分别为 `0.62136/0.64443`、`0.65647/0.67172`、`0.66580/0.67585`、`0.66830/0.67637`；full-support Fourier/Wiener 为 `0.68419/0.68491`。各 target-fitted observer 的 within-class rotation CI 下界均不为正，未达到相对 v61 raw corr/dice 的 90% paired 门。
- v65 预注册决策为 `nonlocal-codebook-objective-representation-mismatch-confirmed`，dense v61 reference 仍可解码。证据排除了 fixed common-mode display、pointwise photometric、`<=31x31` finite-local 和 translation-equivariant Fourier/Wiener 作为充分读出；raw CMOS 人眼不可读 gate 仍失败。该结果只授权进入 generator/objective 架构重设计讨论，不授权直接训练、延长 epoch、改 loss、decoder、D2NN、OLS、aperture、spacing、pupil 或 promotion。

代表证据：

- `tools/diagnose_v65_common_mode_readout_locality_audit.py`
- `outputs/reports/v65_common_mode_readout_locality_audit.json`
- `outputs/reports/v65_common_mode_readout_locality_audit.pt`
- `outputs/reports/v65_common_mode_readout_locality_audit_preview.png`

### v66：semantic-gradient budget audit（CUDA 已完成，diagnostic-only）

- 已新增只读工具 `tools/diagnose_v66_semantic_gradient_budget_audit.py` 与对应测试。工具固定 v60/v56/v58.3、readable `800/100/100` 和 v61–v65 provenance，使用正式未 clamp `camera_plane_loss`、前景阈值 `0.05`、权重 `3.0/1.0`，对 `generator_e`、encoder phase、decoder phase 做逐样本 VJP；不创建训练对象、不写入参数梯度、不保存完整复场或梯度。
- target 只由 train split 构造 global/label template；test residual 的 common、label、instance 分解含精确加和校验，并流式报告参数 RMS、归一化 semantic/common drive、cancellation ratio 及同类/跨类 cosine。所有预注册决策分支均有合成测试。
- v66 已在 CUDA 上按预注册输入运行一次：`parameter_update_count=0`、`optimizer_created=false`，报告、tensor 和 preview 均已生成。结果为 `mixed-semantic-gradient-evidence`：label/instance drive 未同时低于 suppression gate，且未满足 semantic-gradient-available 或 batch-cancellation gate。因此它不授权继续 v56/v60 同配置训练，但允许在明确隔离的理想透明薄相位 diagnostic 分支中进行一次预注册 paired experiment；该分支不得替换 formal contract 或 promotion v60。
- 实现验证已通过：v57–v66 定向 `58 passed`，完整回归 `522 passed`、`148 subtests passed`；compileall、ruff、`git diff --check` 均通过。
